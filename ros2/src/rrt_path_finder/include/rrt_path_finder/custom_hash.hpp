#include <tuple>
#include <functional>

struct TupleHash {
    size_t operator()(const std::tuple<int, int, int>& key) const {
        auto [x, y, z] = key;
        // Combine hashes using boost::hash_combine-like technique
        size_t seed = 0;
        seed ^= std::hash<int>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
