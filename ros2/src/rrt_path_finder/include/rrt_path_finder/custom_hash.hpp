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

struct Vec3dHash {
    double precision; // e.g., 1e-6

    Vec3dHash(double p = 1e-6) : precision(p) {}

    size_t operator()(const Eigen::Vector3d &v) const {
        auto toInt = [&](double x) {
            return static_cast<int>(std::llround(x / precision));
        };
        std::tuple<int,int,int> key(toInt(v.x()), toInt(v.y()), toInt(v.z()));

        // Same combine logic as in your TupleHash
        size_t seed = 0;
        auto [x, y, z] = key;
        seed ^= std::hash<int>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct Vec3dEqual {
    double precision;

    Vec3dEqual(double p = 1e-6) : precision(p) {}

    bool operator()(const Eigen::Vector3d &a, const Eigen::Vector3d &b) const {
        return (std::fabs(a.x() - b.x()) < precision &&
                std::fabs(a.y() - b.y()) < precision &&
                std::fabs(a.z() - b.z()) < precision);
    }
};