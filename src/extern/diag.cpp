#include <cstdint>
#include <cmath>
#include <omp.h>

extern "C"
{
    double build_ritz_and_residual(
        const int64_t N, const int dim, const double *__restrict__ coeffs,
        const double min_e, const double **__restrict__ V, const double **__restrict__ AV,
        double *__restrict__ axt)
    {
        const double *local_V[64];
        const double *local_AV[64];
        double local_c[64];

        for (int j = 0; j < dim; ++j)
        {
            local_V[j] = *(V + j);
            local_AV[j] = *(AV + j);
            local_c[j] = *(coeffs + j);
        }

        double norm_r_sq = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : norm_r_sq)
        for (int64_t i = 0; i < N; ++i)
        {
            double val_x = 0.0;
            double val_ax = 0.0;

#pragma GCC unroll 8
            for (int j = 0; j < dim; ++j)
            {
                const double c = local_c[j];
                val_x += c * *(local_V[j] + i);
                val_ax += c * *(local_AV[j] + i);
            }

            double val_r = val_ax - min_e * val_x;
            *(axt + i) = val_r; // 🌟 用 axt 原地承载残差
            norm_r_sq += val_r * val_r;
        }
        return std::sqrt(norm_r_sq);
    }

    // 2. 新增：只在结束时调用的重组函数
    void build_ritz_vector(
        const int64_t N, const int dim, const double *__restrict__ coeffs,
        const double **__restrict__ V, double *__restrict__ xt)
    {
        const double *local_V[64];
        double local_c[64];
        for (int j = 0; j < dim; ++j)
        {
            local_V[j] = *(V + j);
            local_c[j] = *(coeffs + j);
        }
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i)
        {
            double val_x = 0.0;
#pragma GCC unroll 8
            for (int j = 0; j < dim; ++j)
                val_x += local_c[j] * *(local_V[j] + i);
            *(xt + i) = val_x;
        }
    }

    // =========================================================================
    // 2. 原地预条件子 (In-place Preconditioner)
    // =========================================================================
    void apply_preconditioner_inplace(
        const int64_t N,
        double *__restrict__ axt,
        const double *__restrict__ diags,
        const double min_e,
        const double shift)
    {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i)
        {
            double diff = *(diags + i) - min_e;
            if (std::abs(diff) < shift)
            {
                diff = std::copysign(shift, diff);
            }
            // 🌟 核心：读出当前的残差，除以 diff 后，原地写回自己！
            *(axt + i) = *(axt + i) / diff;
        }
    }
    // =========================================================================
    // 3. MGS 正交化核心 (包含自动归一化)
    // =========================================================================
    double mgs_orthogonalize(
        const int64_t N,
        const int dim,
        const double **__restrict__ V,
        double *__restrict__ vt)
    {
        for (int j = 0; j < dim; ++j)
        {
            const double *__restrict__ vj = *(V + j);
            double proj = 0.0;

// 阶段 1：点积投影
#pragma omp parallel for schedule(static) reduction(+ : proj)
            for (int64_t i = 0; i < N; ++i)
            {
                proj += *(vj + i) * *(vt + i);
            }

// 阶段 2：正交扣除
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < N; ++i)
            {
                *(vt + i) -= proj * *(vj + i);
            }
        }

        // 阶段 3：计算自身的 Norm
        double norm_sq = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : norm_sq)
        for (int64_t i = 0; i < N; ++i)
        {
            const double val = *(vt + i);
            norm_sq += val * val;
        }

        return std::sqrt(norm_sq);
    }

    // =========================================================================
    // 4. 原地归一化 (In-place Normalize)
    // =========================================================================
    double inplace_normalize(
        const int64_t N,
        double *__restrict__ vec)
    {
        double norm_sq = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : norm_sq)
        for (int64_t i = 0; i < N; ++i)
        {
            const double val = *(vec + i);
            norm_sq += val * val;
        }

        const double nrm = std::sqrt(norm_sq);
        const double inv_nrm = 1.0 / nrm;

#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i)
        {
            *(vec + i) *= inv_nrm;
        }

        return nrm;
    }

    // =========================================================================
    // 5. 极速单点积 (彻底摆脱 OpenBLAS 线程争用)
    // =========================================================================
    double fast_dot(const int64_t N, const double *__restrict__ v1, const double *__restrict__ v2)
    {
        double res = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : res)
        for (int64_t i = 0; i < N; ++i)
        {
            res += *(v1 + i) * *(v2 + i);
        }
        return res;
    }

    // =========================================================================
    // 6. 极致融合：子空间扩展 (一次遍历，算出全列 heff)
    // 相当于 Julia 中的: [dot(V[1], AV[dim]), dot(V[2], AV[dim]), ..., dot(V[dim], AV[dim])]
    // =========================================================================
    void compute_heff_col(
        const int64_t N, const int dim,
        const double **__restrict__ V, const double *__restrict__ AV_dim,
        double *__restrict__ out_col)
    {
        // 将指针解构到栈上
        const double *local_V[64];
        for (int j = 0; j < dim; ++j)
        {
            local_V[j] = *(V + j);
            *(out_col + j) = 0.0;
        }

#pragma omp parallel
        {
            // 线程局部累加器，避免竞争
            double local_acc[64] = {0.0};

#pragma omp for schedule(static)
            for (int64_t i = 0; i < N; ++i)
            {
                const double av_val = *(AV_dim + i);

#pragma GCC unroll 8
                for (int j = 0; j < dim; ++j)
                {
                    local_acc[j] += *(local_V[j] + i) * av_val;
                }
            }

// 合并各线程的局部结果
#pragma omp critical
            {
                for (int j = 0; j < dim; ++j)
                {
                    *(out_col + j) += local_acc[j];
                }
            }
        }
    }

    // =========================================================================
    // 7. 极致融合：厚重启重建 Heff (一次遍历，算出整个 heff 矩阵)
    // =========================================================================
    void rebuild_heff(
        const int64_t N, const int dim,
        const double **__restrict__ V, const double **__restrict__ AV,
        double *__restrict__ heff, const int ldh)
    {
        const double *local_V[64];
        const double *local_AV[64];
        for (int j = 0; j < dim; ++j)
        {
            local_V[j] = *(V + j);
            local_AV[j] = *(AV + j);
        }

        // 初始化输出矩阵
        for (int j = 0; j < dim; ++j)
        {
            for (int i = 0; i < dim; ++i)
            {
                *(heff + i + j * ldh) = 0.0;
            }
        }

#pragma omp parallel
        {
            double local_heff[64][64] = {0.0}; // 假设 maxspace 不超过 64

#pragma omp for schedule(static)
            for (int64_t i = 0; i < N; ++i)
            {
#pragma GCC unroll 4
                for (int col = 0; col < dim; ++col)
                {
                    const double av_val = *(local_AV[col] + i);
#pragma GCC unroll 4
                    for (int row = 0; row < dim; ++row)
                    {
                        local_heff[row][col] += *(local_V[row] + i) * av_val;
                    }
                }
            }

#pragma omp critical
            {
                for (int col = 0; col < dim; ++col)
                {
                    for (int row = 0; row < dim; ++row)
                    {
                        *(heff + row + col * ldh) += local_heff[row][col];
                    }
                }
            }
        }
    }

} // extern "C"
