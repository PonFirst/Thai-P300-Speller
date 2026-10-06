////////////////////////////////////////////////////////////////////////////////
// ThaiGPT: the Thai character-level Transformer LM prior, in C++ for BCI2000.
//
// A forward-pass port of stopping/transformer_lm.py (ThaiGPT + ThaiTransformerLM.get_prior),
// reading the weights pipeline/bci2000_lm_export.py writes. No BCI2000 dependencies, so it is
// tested on its own (tests/test_bci2000_stopping.py: priors equal the PyTorch model's).
//
// Architecture, as gpt_dev.ipynb trained it: token + position embeddings; n_layer blocks of
//   x += proj(concat_h softmax(mask(q_h k_h^T * n_embd^-0.5)) v_h)   on LayerNorm(x)
//   x += W2 relu(W1 LayerNorm(x) + b1) + b2
// then LayerNorm and the vocabulary head. Note the attention scale is n_embd^-0.5, not
// head_size^-0.5: that is how the model was trained, so it is reproduced exactly.
// get_prior: the context's last block_size characters (characters outside the vocabulary map to
// <unk>; an empty context is a single <unk>), softmax over the whole vocabulary at the last
// position, the matrix characters' probabilities + 1e-8, renormalised.
// Computation is float32 like PyTorch's CPU forward; the prior is assembled in double.
////////////////////////////////////////////////////////////////////////////////
#ifndef THAI_GPT_H
#define THAI_GPT_H

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ThaiGPT
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;

// UTF-8 to code points; an invalid byte becomes U+FFFD (not in the vocabulary, so it maps to <unk>).
inline std::vector<char32_t> DecodeUtf8(const std::string &s)
{
    std::vector<char32_t> out;
    for (size_t i = 0; i < s.size();)
    {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size())
        {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        bool ok = true;
        for (int k = 1; k < n; ++k)
        {
            unsigned char d = static_cast<unsigned char>(s[i + k]);
            ok = ok && (d >> 6) == 2;
            cp = (cp << 6) | (d & 0x3F);
        }
        out.push_back(ok ? cp : 0xFFFD);
        i += ok ? n : 1;
    }
    return out;
}

struct Layer
{
    Vec ln1w, ln1b, ln2w, ln2b, projB, ff1B, ff2B;
    std::vector<Mat> key, query, value; // per head (headSize, C)
    Mat projW, ff1W, ff2W;
};

class Model
{
  public:
    int vocabSize = 0, blockSize = 0, nEmbd = 0, nHead = 0, nLayer = 0, unkId = 0;
    std::vector<char32_t> vocab, matrix;
    std::string checkpointSha;

    // Empty on success, else what is wrong with the file.
    std::string Load(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return "cannot open " + path;
        std::string line;
        if (!std::getline(f, line) || line != "THAIGPT 1")
            return path + " is not a Thai GPT export (pipeline/bci2000_lm_export.py)";
        if (!std::getline(f, line))
            return "truncated header";
        std::istringstream dims(line);
        if (!(dims >> vocabSize >> blockSize >> nEmbd >> nHead >> nLayer >> unkId) || vocabSize < 1 || blockSize < 1 ||
            nEmbd < 1 || nHead < 1 || nEmbd % nHead || nLayer < 1 || unkId < 0 || unkId >= vocabSize)
            return "bad model dimensions";
        if (!std::getline(f, line))
            return "truncated header";
        vocab = ParseHex(line);
        if (static_cast<int>(vocab.size()) != vocabSize)
            return "vocabulary length does not match vocab_size";
        stoi_.clear();
        for (int i = 0; i < vocabSize; ++i)
            stoi_[vocab[i]] = i;
        if (!std::getline(f, line) || line.rfind("checkpoint ", 0) != 0)
            return "missing checkpoint line";
        checkpointSha = line.substr(11);
        if (!std::getline(f, line) || line.rfind("matrix ", 0) != 0)
            return "missing matrix line";
        matrix = ParseHex(line.substr(7));
        if (!std::getline(f, line) || line != "DATA")
            return "missing DATA line";

        const int C = nEmbd, hs = nEmbd / nHead;
        bool ok = Read(f, tok_, vocabSize, C) && Read(f, pos_, blockSize, C);
        layers_.assign(nLayer, Layer());
        for (Layer &l : layers_)
        {
            ok = ok && Read(f, l.ln1w, C) && Read(f, l.ln1b, C);
            l.key.resize(nHead);
            l.query.resize(nHead);
            l.value.resize(nHead);
            for (int h = 0; h < nHead; ++h)
                ok = ok && Read(f, l.key[h], hs, C) && Read(f, l.query[h], hs, C) && Read(f, l.value[h], hs, C);
            ok = ok && Read(f, l.projW, C, C) && Read(f, l.projB, C) && Read(f, l.ln2w, C) && Read(f, l.ln2b, C) &&
                 Read(f, l.ff1W, 4 * C, C) && Read(f, l.ff1B, 4 * C) && Read(f, l.ff2W, C, 4 * C) && Read(f, l.ff2B, C);
        }
        ok = ok && Read(f, lnfW_, C) && Read(f, lnfB_, C) && Read(f, headW_, vocabSize, C) && Read(f, headB_, vocabSize);
        if (!ok)
            return "truncated weights";
        if (f.peek() != EOF)
            return "trailing bytes after the weights";
        return "";
    }

    // The token id of a code point, -1 if it is not in the vocabulary.
    int Id(char32_t c) const
    {
        auto it = stoi_.find(c);
        return it == stoi_.end() ? -1 : it->second;
    }

    // ThaiTransformerLM.get_prior(context) over the characters whose token ids are matrixIds.
    std::vector<double> Prior(const std::string &contextUtf8, const std::vector<int> &matrixIds) const
    {
        std::vector<int> ids;
        for (char32_t c : DecodeUtf8(contextUtf8))
        {
            int id = Id(c);
            ids.push_back(id < 0 ? unkId : id);
        }
        if (static_cast<int>(ids.size()) > blockSize)
            ids.erase(ids.begin(), ids.end() - blockSize);
        if (ids.empty())
            ids.push_back(unkId);
        Vec logits = LastLogits(ids);
        const float m = logits.maxCoeff();
        Eigen::ArrayXf e = (logits.array() - m).exp();
        Eigen::ArrayXf probs = e / e.sum();
        std::vector<double> raw(matrixIds.size());
        double total = 0;
        for (size_t i = 0; i < matrixIds.size(); ++i)
        {
            raw[i] = static_cast<double>(probs(matrixIds[i])) + 1e-8;
            total += raw[i];
        }
        for (double &r : raw)
            r /= total;
        return raw;
    }

  private:
    static std::vector<char32_t> ParseHex(const std::string &line)
    {
        std::vector<char32_t> out;
        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok)
            out.push_back(static_cast<char32_t>(std::stoul(tok, nullptr, 16)));
        return out;
    }
    static bool Read(std::ifstream &f, Mat &m, int rows, int cols)
    {
        m.resize(rows, cols);
        return static_cast<bool>(f.read(reinterpret_cast<char *>(m.data()), sizeof(float) * rows * cols));
    }
    static bool Read(std::ifstream &f, Vec &v, int n)
    {
        v.resize(n);
        return static_cast<bool>(f.read(reinterpret_cast<char *>(v.data()), sizeof(float) * n));
    }
    static Mat LayerNorm(const Mat &x, const Vec &w, const Vec &b)
    {
        Mat out(x.rows(), x.cols());
        for (int r = 0; r < x.rows(); ++r)
        {
            const float mean = x.row(r).mean();
            const float var = (x.row(r).array() - mean).square().mean();
            out.row(r) = (((x.row(r).array() - mean) / std::sqrt(var + 1e-5f)) * w.transpose().array() +
                          b.transpose().array())
                             .matrix();
        }
        return out;
    }

    Vec LastLogits(const std::vector<int> &ids) const
    {
        const int T = static_cast<int>(ids.size()), C = nEmbd, hs = nEmbd / nHead;
        const float scale = 1.0f / std::sqrt(static_cast<float>(C)); // n_embd^-0.5, as trained
        Mat x(T, C);
        for (int t = 0; t < T; ++t)
            x.row(t) = tok_.row(ids[t]) + pos_.row(t);
        for (const Layer &l : layers_)
        {
            Mat h = LayerNorm(x, l.ln1w, l.ln1b);
            Mat cat(T, C);
            for (int k = 0; k < nHead; ++k)
            {
                Mat q = h * l.query[k].transpose(), kk = h * l.key[k].transpose(), v = h * l.value[k].transpose();
                Mat wei = (q * kk.transpose()) * scale;
                for (int i = 0; i < T; ++i)
                {
                    for (int j = i + 1; j < T; ++j)
                        wei(i, j) = -INFINITY;
                    const float m = wei.row(i).head(i + 1).maxCoeff();
                    Eigen::ArrayXf e = (wei.row(i).head(i + 1).array() - m).exp();
                    wei.row(i).head(i + 1) = (e / e.sum()).matrix().transpose();
                    wei.row(i).tail(T - i - 1).setZero();
                }
                cat.middleCols(k * hs, hs) = wei * v;
            }
            x += (cat * l.projW.transpose()).rowwise() + l.projB.transpose();
            Mat h2 = LayerNorm(x, l.ln2w, l.ln2b);
            Mat ff = ((h2 * l.ff1W.transpose()).rowwise() + l.ff1B.transpose()).cwiseMax(0.0f);
            x += (ff * l.ff2W.transpose()).rowwise() + l.ff2B.transpose();
        }
        Mat last = LayerNorm(x.bottomRows(1), lnfW_, lnfB_);
        return headW_ * last.row(0).transpose() + headB_;
    }

    std::unordered_map<char32_t, int> stoi_;
    Mat tok_, pos_, headW_;
    Vec lnfW_, lnfB_, headB_;
    std::vector<Layer> layers_;
};

} // namespace ThaiGPT

#endif // THAI_GPT_H
