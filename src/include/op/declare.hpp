#pragma once
#include "ham/otf.hpp"
#include "op/hvec.hpp"
#include "op/tvec.hpp"
#include "op/expm.hpp"
#include "op/grad.hpp"
#include "op/backgrad.hpp"
#include "op/backtran.hpp"
#include "op/batchexpm.hpp"
#include "op/batchgrad.hpp"
#include "op/batchtran.hpp"

#define DECLARE_OP_INTERFACES(Ti, Tv, SUFFIX)                                                   \
    extern "C"                                                                                  \
    {                                                                                           \
        void destroy_network_otf##SUFFIX(void *net_ptr)                                         \
        {                                                                                       \
            if (net_ptr == nullptr)                                                             \
                return;                                                                         \
            auto *net = static_cast<Network_OTF<Ti, Tv> *>(net_ptr);                            \
            net->clear();                                                                       \
            delete net;                                                                         \
        }                                                                                       \
                                                                                                \
        void *build_network_otf##SUFFIX(                                                        \
            const int64 *orbsym, int64 norb, int64 ngs,                                         \
            const Ti *axs, const Ti *bxs,                                                       \
            const int64 *ranks, const int64 *num_zas, const int64 *num_zbs,                     \
            const Ti *flat_zas, const Ti *flat_zbs,                                             \
            const Tv *flat_wa, const Tv *flat_wb)                                               \
        {                                                                                       \
            return build_network_otf<Ti, Tv>(                                                   \
                orbsym, norb, ngs, axs, bxs, ranks, num_zas, num_zbs,                           \
                flat_zas, flat_zbs, flat_wa, flat_wb);                                          \
        }                                                                                       \
                                                                                                \
        void hvec_gather_contract_otf##SUFFIX(                                                  \
            void *basis_ptr, void *net_ptr,                                                     \
            const Tv *__restrict__ src, Tv *__restrict__ dst)                                   \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            contract_network_otf<Ti, Tv>(basis, net, src, dst);                                 \
        }                                                                                       \
                                                                                                \
        void expm_contract_otf##SUFFIX(                                                         \
            void *basis_ptr, void *net_ptr,                                                     \
            const int64 idx, const double theta,                                                \
            Tv *__restrict__ vec)                                                               \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            expm_svd_network_otf<Ti, Tv>(basis, net, idx, theta, vec);                          \
        }                                                                                       \
                                                                                                \
        Tv grad_contract_otf##SUFFIX(                                                           \
            void *basis_ptr, void *net_ptr,                                                     \
            const int64 idx, const double theta,                                                \
            const Tv *__restrict__ lp, const Tv *__restrict__ rp)                               \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            return grad_svd_network_otf<Ti, Tv>(basis, net, idx, theta, lp, rp);                \
        }                                                                                       \
                                                                                                \
        Tv backgrad_contract_otf##SUFFIX(                                                       \
            void *basis_ptr, void *net_ptr,                                                     \
            const int64 idx, const double theta,                                                \
            Tv *__restrict__ lp, Tv *__restrict__ rp)                                           \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            return backgrad_svd_network_otf<Ti, Tv>(basis, net, idx, theta, lp, rp);            \
        }                                                                                       \
                                                                                                \
        void backtran_contract_otf##SUFFIX(                                                     \
            void *basis_ptr, void *net_ptr,                                                     \
            const int64 idx, const double theta,                                                \
            Tv *__restrict__ lp, Tv *__restrict__ rp, Tv *__restrict__ bp)                      \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            backtran_svd_network_otf<Ti, Tv>(basis, net, idx, theta, lp, rp, bp);               \
        }                                                                                       \
                                                                                                \
        void tvec_contract_otf##SUFFIX(                                                         \
            void *basis_ptr, void *net_ptr,                                                     \
            const int64 idx, const Tv *__restrict__ src, Tv *__restrict__ dst)                  \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            tvec_svd_network_otf<Ti, Tv>(basis, net, idx, src, dst);                            \
        }                                                                                       \
                                                                                                \
        void batch_expm_contract_otf##SUFFIX(                                                   \
            void *basis_ptr, void *net_ptr,                                                     \
            const int64 idx, const double theta,                                                \
            Tv *__restrict__ matrix, int ld, int num_vecs)                                      \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            expm_svd_batched_network_otf<Ti, Tv>(basis, net, idx, theta, matrix, ld, num_vecs); \
        }                                                                                       \
                                                                                                \
        void batch_grad_contract_otf##SUFFIX(                                                   \
            void *basis_ptr, void *net_ptr,                                                     \
            const double *__restrict__ thetas,                                                  \
            const Tv *__restrict__ lp, const Tv *__restrict__ rp,                               \
            Tv *__restrict__ grads)                                                             \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            grad_pool_network_batched_otf<Ti, Tv>(basis, net, thetas, lp, rp, grads);           \
        }                                                                                       \
                                                                                                \
        void batch_tran_contract_otf##SUFFIX(                                                   \
            void *basis_ptr, void *net_ptr,                                                     \
            const Tv *__restrict__ lp, const Tv *__restrict__ rp,                               \
            Tv *__restrict__ trans)                                                             \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            tran_pool_network_batched_otf<Ti, Tv>(basis, net, lp, rp, trans);                   \
        }                                                                                       \
                                                                                                \
        void get_diags_elements##SUFFIX(void *basis_ptr, void *net_ptr, Tv *diags)              \
        {                                                                                       \
            auto *basis = static_cast<const BasisManager<Ti> *>(basis_ptr);                     \
            auto *net = static_cast<const Network_OTF<Ti, Tv> *>(net_ptr);                      \
            get_diags_elements<Ti, Tv>(basis, net, diags);                                      \
        }                                                                                       \
    }
