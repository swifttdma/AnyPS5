#include "LoweringExecution.hpp"
#include "VectorExecution.hpp"
#include <codegen/x86/FmaLowering.hpp>
#include <codegen/x86/IAmd64OnlyInstructionMatcher.hpp>
#include <codegen/x86/X64InstructionDecoder.hpp>
#include <codegen/CodegenException.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using LoweringExecution::Bytes;
using Vector = std::array<std::uint64_t, 4>;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void append32(Bytes& bytes, const std::int32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(static_cast<std::uint32_t>(value) >> shift));
}

Bytes vexFmaReg(const std::uint8_t opcode, const bool wideW, const bool vec256, const std::uint8_t reg, const std::uint8_t vvvv, const std::uint8_t rm) {
    return {
        0xC4,
        static_cast<std::uint8_t>(((reg & 8) ? 0 : 0x80) | 0x40 | ((rm & 8) ? 0 : 0x20) | 2),
        static_cast<std::uint8_t>((wideW ? 0x80 : 0) | ((~vvvv & 15) << 3) | (vec256 ? 4 : 0) | 1),
        opcode,
        static_cast<std::uint8_t>(0xC0 | ((reg & 7) << 3) | (rm & 7)),
    };
}

Bytes vexFmaMemBase(const std::uint8_t opcode, const bool wideW, const bool vec256, const std::uint8_t reg, const std::uint8_t vvvv, const std::uint8_t base) {
    Bytes out{
        0xC4,
        static_cast<std::uint8_t>(((reg & 8) ? 0 : 0x80) | 0x40 | ((base & 8) ? 0 : 0x20) | 2),
        static_cast<std::uint8_t>((wideW ? 0x80 : 0) | ((~vvvv & 15) << 3) | (vec256 ? 4 : 0) | 1),
        opcode,
    };
    if ((base & 7) == 5) {
        out.push_back(static_cast<std::uint8_t>(0x40 | ((reg & 7) << 3) | 5));
        out.push_back(0);
    } else if ((base & 7) == 4) {
        out.push_back(static_cast<std::uint8_t>(((reg & 7) << 3) | 4));
        out.push_back(0x24);
    } else {
        out.push_back(static_cast<std::uint8_t>(((reg & 7) << 3) | (base & 7)));
    }
    return out;
}

Bytes vexFmaRip(const std::uint8_t opcode, const bool wideW, const bool vec256, const std::uint8_t reg, const std::uint8_t vvvv, const std::int32_t disp) {
    Bytes out{
        0xC4,
        static_cast<std::uint8_t>(((reg & 8) ? 0 : 0x80) | 0x60 | 2),
        static_cast<std::uint8_t>((wideW ? 0x80 : 0) | ((~vvvv & 15) << 3) | (vec256 ? 4 : 0) | 1),
        opcode,
        static_cast<std::uint8_t>(((reg & 7) << 3) | 5),
    };
    append32(out, disp);
    return out;
}

Bytes vexFmaSib(const std::uint8_t opcode, const bool wideW, const bool vec256, const std::uint8_t reg, const std::uint8_t vvvv, const std::uint8_t base, const std::uint8_t index, const std::uint8_t scaleBits, const std::int32_t disp) {
    Bytes out{
        0xC4,
        static_cast<std::uint8_t>(((reg & 8) ? 0 : 0x80) | ((index & 8) ? 0 : 0x40) | ((base & 8) ? 0 : 0x20) | 2),
        static_cast<std::uint8_t>((wideW ? 0x80 : 0) | ((~vvvv & 15) << 3) | (vec256 ? 4 : 0) | 1),
        opcode,
        static_cast<std::uint8_t>(0x84 | ((reg & 7) << 3)),
        static_cast<std::uint8_t>((scaleBits << 6) | ((index & 7) << 3) | (base & 7)),
    };
    append32(out, disp);
    return out;
}

Bytes vexFmaRspDisp(const std::uint8_t opcode, const bool wideW, const bool vec256, const std::uint8_t reg, const std::uint8_t vvvv, const std::int32_t disp) {
    Bytes out{
        0xC4,
        static_cast<std::uint8_t>(((reg & 8) ? 0 : 0x80) | 0x60 | 2),
        static_cast<std::uint8_t>((wideW ? 0x80 : 0) | ((~vvvv & 15) << 3) | (vec256 ? 4 : 0) | 1),
        opcode,
        static_cast<std::uint8_t>(0x84 | ((reg & 7) << 3)),
        0x24,
    };
    append32(out, disp);
    return out;
}

int bsr32Ref(std::uint32_t value) {
    int bit = 31;
    while (bit > 0 && ((value >> bit) & 1u) == 0) --bit;
    return bit;
}

int bsr64Ref(std::uint64_t value) {
    int bit = 63;
    while (bit > 0 && ((value >> bit) & 1ull) == 0) --bit;
    return bit;
}

std::uint32_t sdmFma32(std::uint32_t a, std::uint32_t b, std::uint32_t c, const bool negProd, const bool negC, const std::uint32_t mxcsr) {
    const bool daz = (mxcsr & 0x0040u) != 0;
    const bool ftz = (mxcsr & 0x8000u) != 0;
    const std::uint32_t rc = (mxcsr >> 13) & 3u;
    const auto flushDaz = [&](std::uint32_t x) {
        return daz && (x & 0x7F800000u) == 0 ? (x & 0x80000000u) : x;
    };
    a = flushDaz(a);
    b = flushDaz(b);
    c = flushDaz(c);
    const auto isNan = [](std::uint32_t x) { return (x & 0x7FFFFFFFu) > 0x7F800000u; };
    if (isNan(a)) return a | 0x00400000u;
    if (isNan(b)) return b | 0x00400000u;
    if (isNan(c)) return c | 0x00400000u;
    const std::uint32_t sp = ((a ^ b) >> 31) ^ (negProd ? 1u : 0u);
    const std::uint32_t sc = (c >> 31) ^ (negC ? 1u : 0u);
    const std::uint32_t aa = a & 0x7FFFFFFFu;
    const std::uint32_t ab = b & 0x7FFFFFFFu;
    const std::uint32_t ac = c & 0x7FFFFFFFu;
    const bool aInf = aa == 0x7F800000u;
    const bool bInf = ab == 0x7F800000u;
    const bool cInf = ac == 0x7F800000u;
    if ((aInf && ab == 0) || (bInf && aa == 0)) return 0xFFC00000u;
    if (aInf || bInf) {
        if (cInf && sp != sc) return 0xFFC00000u;
        return (sp << 31) | 0x7F800000u;
    }
    if (cInf) return (sc << 31) | 0x7F800000u;
    const bool pZero = aa == 0 || ab == 0;
    if (pZero && ac == 0) {
        const std::uint32_t sign = sp == sc ? sp : (rc == 1u ? 1u : 0u);
        return sign << 31;
    }
    const auto unpack = [](std::uint32_t bits, std::int32_t& exp, std::uint64_t& mant) {
        const std::uint32_t e = bits >> 23;
        const std::uint32_t f = bits & 0x7FFFFFu;
        if (e != 0) {
            exp = static_cast<std::int32_t>(e) - 127;
            mant = (1ull << 23) | f;
        } else {
            const int msb = bsr32Ref(f);
            exp = -149 + msb;
            mant = static_cast<std::uint64_t>(f) << (23 - msb);
        }
    };
    std::uint64_t w1 = 0;
    std::uint64_t w0 = 0;
    std::int32_t exp = 0;
    std::uint32_t sign = 0;
    if (pZero) {
        std::uint64_t mc = 0;
        unpack(ac, exp, mc);
        w1 = mc << 38;
        sign = sc;
    } else {
        std::uint64_t ma = 0;
        std::uint64_t mb = 0;
        std::int32_t ea = 0;
        std::int32_t eb = 0;
        unpack(aa, ea, ma);
        unpack(ab, eb, mb);
        std::uint64_t mp = ma * mb;
        std::int32_t ep = ea + eb;
        if ((mp & (1ull << 47)) != 0) ++ep;
        else mp <<= 1;
        if (ac == 0) {
            w1 = mp << 14;
            exp = ep;
            sign = sp;
        } else {
            std::uint64_t mc24 = 0;
            std::int32_t ec = 0;
            unpack(ac, ec, mc24);
            std::uint64_t mc = mc24 << 24;
            std::uint64_t big = mp;
            std::uint64_t small = mc;
            std::int32_t ebig = ep;
            std::int32_t esmall = ec;
            std::uint32_t sbig = sp;
            std::uint32_t ssmall = sc;
            if (ec > ep || (ec == ep && mc > mp)) {
                std::swap(big, small);
                std::swap(ebig, esmall);
                std::swap(sbig, ssmall);
            }
            sign = sbig;
            exp = ebig;
            std::int32_t diff = ebig - esmall;
            std::uint64_t b1 = big << 14;
            std::uint64_t b0 = 0;
            std::uint64_t sm1 = small << 14;
            std::uint64_t sm0 = 0;
            if (diff >= 127) {
                sm1 = 0;
                sm0 = 1;
            } else {
                std::uint64_t dropped = 0;
                if (diff >= 64) {
                    dropped = sm0;
                    sm0 = sm1;
                    sm1 = 0;
                    diff -= 64;
                }
                if (diff > 0) {
                    dropped |= sm0 << (64 - diff);
                    sm0 = (sm0 >> diff) | (sm1 << (64 - diff));
                    sm1 >>= diff;
                }
                if (dropped != 0) sm0 |= 1;
            }
            if (sbig == ssmall) {
                w0 = b0 + sm0;
                w1 = b1 + sm1 + (w0 < b0 ? 1ull : 0ull);
            } else {
                w0 = b0 - sm0;
                w1 = b1 - sm1 - (b0 < sm0 ? 1ull : 0ull);
                if (w1 == 0 && w0 == 0) return rc == 1u ? 0x80000000u : 0u;
            }
        }
    }
    if (w1 == 0) {
        w1 = w0;
        w0 = 0;
        exp -= 64;
    }
    const int msb = bsr64Ref(w1);
    exp += msb - 61;
    const int lsh = 63 - msb;
    if (lsh > 0) {
        w1 = (w1 << lsh) | (w0 >> (64 - lsh));
        w0 <<= lsh;
    }
    if (w0 != 0) w1 |= 1;
    bool subnormal = false;
    if (exp < -126 && !ftz) {
        subnormal = true;
        const std::int32_t rsh = -126 - exp;
        if (rsh >= 26) {
            w1 = 1;
        } else {
            const std::uint64_t dropped = w1 << (64 - rsh);
            w1 >>= rsh;
            if (dropped != 0) w1 |= 1;
        }
    }
    std::uint64_t sig = w1 >> 40;
    const std::uint32_t rbit = static_cast<std::uint32_t>((w1 >> 39) & 1ull);
    const bool sticky = (w1 & ((1ull << 39) - 1)) != 0;
    bool inc = false;
    if (rc == 0) inc = rbit != 0 && (sticky || (sig & 1ull) != 0);
    else if (rc == 1) inc = sign != 0 && (rbit != 0 || sticky);
    else if (rc == 2) inc = sign == 0 && (rbit != 0 || sticky);
    if (inc) ++sig;
    if (subnormal) return (sign << 31) | static_cast<std::uint32_t>(sig);
    if (sig == (1ull << 24)) {
        sig >>= 1;
        ++exp;
    }
    if (exp < -126) return sign << 31;
    if (exp > 127) {
        const bool toInf = rc == 0 || (rc == 1 && sign != 0) || (rc == 2 && sign == 0);
        return (sign << 31) | (toInf ? 0x7F800000u : 0x7F7FFFFFu);
    }
    return (sign << 31) | (static_cast<std::uint32_t>(exp + 127) << 23) | static_cast<std::uint32_t>(sig & 0x7FFFFFu);
}

void mul64x64(const std::uint64_t a, const std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo) {
    const std::uint64_t aLo = a & 0xFFFFFFFFull;
    const std::uint64_t aHi = a >> 32;
    const std::uint64_t bLo = b & 0xFFFFFFFFull;
    const std::uint64_t bHi = b >> 32;
    const std::uint64_t p0 = aLo * bLo;
    const std::uint64_t p1 = aLo * bHi;
    const std::uint64_t p2 = aHi * bLo;
    const std::uint64_t p3 = aHi * bHi;
    const std::uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p2 & 0xFFFFFFFFull);
    lo = (p0 & 0xFFFFFFFFull) | (mid << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}

std::uint64_t sdmFma64(std::uint64_t a, std::uint64_t b, std::uint64_t c, const bool negProd, const bool negC, const std::uint32_t mxcsr) {
    const bool daz = (mxcsr & 0x0040u) != 0;
    const bool ftz = (mxcsr & 0x8000u) != 0;
    const std::uint32_t rc = (mxcsr >> 13) & 3u;
    const auto flushDaz = [&](std::uint64_t x) {
        return daz && (x & 0x7FF0000000000000ull) == 0 ? (x & 0x8000000000000000ull) : x;
    };
    a = flushDaz(a);
    b = flushDaz(b);
    c = flushDaz(c);
    const auto isNan = [](std::uint64_t x) { return (x & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull; };
    if (isNan(a)) return a | 0x0008000000000000ull;
    if (isNan(b)) return b | 0x0008000000000000ull;
    if (isNan(c)) return c | 0x0008000000000000ull;
    const std::uint64_t sp = ((a ^ b) >> 63) ^ (negProd ? 1ull : 0ull);
    const std::uint64_t sc = (c >> 63) ^ (negC ? 1ull : 0ull);
    const std::uint64_t aa = a & 0x7FFFFFFFFFFFFFFFull;
    const std::uint64_t ab = b & 0x7FFFFFFFFFFFFFFFull;
    const std::uint64_t ac = c & 0x7FFFFFFFFFFFFFFFull;
    const bool aInf = aa == 0x7FF0000000000000ull;
    const bool bInf = ab == 0x7FF0000000000000ull;
    const bool cInf = ac == 0x7FF0000000000000ull;
    if ((aInf && ab == 0) || (bInf && aa == 0)) return 0xFFF8000000000000ull;
    if (aInf || bInf) {
        if (cInf && sp != sc) return 0xFFF8000000000000ull;
        return (sp << 63) | 0x7FF0000000000000ull;
    }
    if (cInf) return (sc << 63) | 0x7FF0000000000000ull;
    const bool pZero = aa == 0 || ab == 0;
    if (pZero && ac == 0) {
        const std::uint64_t sign = sp == sc ? sp : (rc == 1u ? 1ull : 0ull);
        return sign << 63;
    }
    const auto unpack = [](std::uint64_t bits, std::int32_t& exp, std::uint64_t& mant) {
        const std::uint64_t e = bits >> 52;
        const std::uint64_t f = bits & 0x000FFFFFFFFFFFFFull;
        if (e != 0) {
            exp = static_cast<std::int32_t>(e) - 1023;
            mant = (1ull << 52) | f;
        } else {
            const int msb = bsr64Ref(f);
            exp = -1074 + msb;
            mant = f << (52 - msb);
        }
    };
    std::uint64_t w2 = 0;
    std::uint64_t w1 = 0;
    std::uint64_t w0 = 0;
    std::int32_t exp = 0;
    std::uint64_t sign = 0;
    if (pZero) {
        std::uint64_t mc = 0;
        unpack(ac, exp, mc);
        w2 = mc << 9;
        sign = sc;
    } else {
        std::uint64_t ma = 0;
        std::uint64_t mb = 0;
        std::int32_t ea = 0;
        std::int32_t eb = 0;
        unpack(aa, ea, ma);
        unpack(ab, eb, mb);
        std::uint64_t pHi = 0;
        std::uint64_t pLo = 0;
        mul64x64(ma, mb, pHi, pLo);
        std::int32_t ep = ea + eb;
        if ((pHi & (1ull << 41)) != 0) {
            ++ep;
        } else {
            pHi = (pHi << 1) | (pLo >> 63);
            pLo <<= 1;
        }
        std::uint64_t bw2 = (pHi << 20) | (pLo >> 44);
        std::uint64_t bw1 = pLo << 20;
        if (ac == 0) {
            w2 = bw2;
            w1 = bw1;
            exp = ep;
            sign = sp;
        } else {
            std::uint64_t mc = 0;
            std::int32_t ec = 0;
            unpack(ac, ec, mc);
            std::uint64_t sw2 = mc << 9;
            std::uint64_t sw1 = 0;
            std::uint64_t sw0 = 0;
            std::uint64_t bw0 = 0;
            std::int32_t ebig = ep;
            std::int32_t esmall = ec;
            std::uint64_t sbig = sp;
            std::uint64_t ssmall = sc;
            if (ec > ep || (ec == ep && sw2 > bw2)) {
                std::swap(bw2, sw2);
                std::swap(bw1, sw1);
                std::swap(ebig, esmall);
                std::swap(sbig, ssmall);
            }
            sign = sbig;
            exp = ebig;
            std::int32_t diff = ebig - esmall;
            if (diff >= 191) {
                sw2 = 0;
                sw1 = 0;
                sw0 = 1;
            } else {
                std::uint64_t dropped = 0;
                if (diff >= 64) {
                    dropped |= sw0;
                    sw0 = sw1;
                    sw1 = sw2;
                    sw2 = 0;
                    diff -= 64;
                }
                if (diff >= 64) {
                    dropped |= sw0;
                    sw0 = sw1;
                    sw1 = 0;
                    diff -= 64;
                }
                if (diff > 0) {
                    dropped |= sw0 << (64 - diff);
                    sw0 = (sw0 >> diff) | (sw1 << (64 - diff));
                    sw1 = (sw1 >> diff) | (sw2 << (64 - diff));
                    sw2 >>= diff;
                }
                if (dropped != 0) sw0 |= 1;
            }
            if (sbig == ssmall) {
                w0 = bw0 + sw0;
                const std::uint64_t c0 = w0 < bw0 ? 1ull : 0ull;
                const std::uint64_t t1 = bw1 + sw1;
                w1 = t1 + c0;
                const std::uint64_t c1 = (t1 < bw1 || w1 < t1) ? 1ull : 0ull;
                w2 = bw2 + sw2 + c1;
            } else {
                const std::uint64_t r0 = bw0 < sw0 ? 1ull : 0ull;
                w0 = bw0 - sw0;
                const std::uint64_t r1 = (bw1 < sw1 || (bw1 == sw1 && r0 != 0)) ? 1ull : 0ull;
                w1 = bw1 - sw1 - r0;
                w2 = bw2 - sw2 - r1;
                if ((w2 | w1 | w0) == 0) return rc == 1u ? 0x8000000000000000ull : 0ull;
            }
        }
    }
    if (w2 == 0) {
        if (w1 != 0) {
            w2 = w1;
            w1 = w0;
            w0 = 0;
            exp -= 64;
        } else {
            w2 = w0;
            w1 = 0;
            w0 = 0;
            exp -= 128;
        }
    }
    const int msb = bsr64Ref(w2);
    exp += msb - 61;
    const int lsh = 63 - msb;
    if (lsh > 0) {
        w2 = (w2 << lsh) | (w1 >> (64 - lsh));
        w1 = (w1 << lsh) | (w0 >> (64 - lsh));
        w0 <<= lsh;
    }
    if ((w1 | w0) != 0) w2 |= 1;
    bool subnormal = false;
    if (exp < -1022 && !ftz) {
        subnormal = true;
        const std::int32_t rsh = -1022 - exp;
        if (rsh >= 55) {
            w2 = 1;
        } else {
            const std::uint64_t dropped = w2 << (64 - rsh);
            w2 >>= rsh;
            if (dropped != 0) w2 |= 1;
        }
    }
    std::uint64_t sig = w2 >> 11;
    const std::uint64_t rbit = (w2 >> 10) & 1ull;
    const bool sticky = (w2 & 0x3FFull) != 0;
    bool inc = false;
    if (rc == 0) inc = rbit != 0 && (sticky || (sig & 1ull) != 0);
    else if (rc == 1) inc = sign != 0 && (rbit != 0 || sticky);
    else if (rc == 2) inc = sign == 0 && (rbit != 0 || sticky);
    if (inc) ++sig;
    if (subnormal) return (sign << 63) | sig;
    if (sig == (1ull << 53)) {
        sig >>= 1;
        ++exp;
    }
    if (exp < -1022) return sign << 63;
    if (exp > 1023) {
        const bool toInf = rc == 0 || (rc == 1 && sign != 0) || (rc == 2 && sign == 0);
        return (sign << 63) | (toInf ? 0x7FF0000000000000ull : 0x7FEFFFFFFFFFFFFFull);
    }
    return (sign << 63) | (static_cast<std::uint64_t>(exp + 1023) << 52) | (sig & 0x000FFFFFFFFFFFFFull);
}

Vector sdmFmaVector(const Codegen::FmaOperands& op, const Vector& v1, const Vector& v2, const Vector& v3, const std::uint32_t mxcsr) {
    std::array<std::uint8_t, 32> b1{};
    std::array<std::uint8_t, 32> b2{};
    std::array<std::uint8_t, 32> b3{};
    std::memcpy(b1.data(), v1.data(), 32);
    std::memcpy(b2.data(), v2.data(), 32);
    std::memcpy(b3.data(), v3.data(), 32);
    std::array<std::uint8_t, 32> res{};
    if (op.Scalar) std::memcpy(res.data(), b1.data(), 16);
    const std::size_t elemBytes = op.DoublePrecision ? 8 : 4;
    const std::size_t lanes = op.Scalar ? 1 : (op.Vector256 ? 32 : 16) / elemBytes;
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        const auto pick = [&](const std::array<std::uint8_t, 32>& src) {
            std::uint64_t val = 0;
            std::memcpy(&val, src.data() + lane * elemBytes, elemBytes);
            return val;
        };
        std::uint64_t a = 0;
        std::uint64_t b = 0;
        std::uint64_t c = 0;
        if (op.Order == Codegen::FmaOrder::Order132) {
            a = pick(b1); b = pick(b3); c = pick(b2);
        } else if (op.Order == Codegen::FmaOrder::Order213) {
            a = pick(b2); b = pick(b1); c = pick(b3);
        } else {
            a = pick(b2); b = pick(b3); c = pick(b1);
        }
        bool negProd = op.Operation == Codegen::FmaOperation::Nmadd || op.Operation == Codegen::FmaOperation::Nmsub;
        bool negC = op.Operation == Codegen::FmaOperation::Msub || op.Operation == Codegen::FmaOperation::Nmsub;
        if (op.Operation == Codegen::FmaOperation::MaddSub) negC = (lane & 1u) == 0;
        if (op.Operation == Codegen::FmaOperation::MsubAdd) negC = (lane & 1u) != 0;
        if (op.DoublePrecision) {
            const auto out = sdmFma64(a, b, c, negProd, negC, mxcsr);
            std::memcpy(res.data() + lane * 8, &out, 8);
        } else {
            const auto out = sdmFma32(static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(c), negProd, negC, mxcsr);
            std::memcpy(res.data() + lane * 4, &out, 4);
        }
    }
    Vector out{};
    std::memcpy(out.data(), res.data(), 32);
    return out;
}

void decoding() {
    constexpr std::array<std::uint8_t, 30> kOpcodes{
        0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF,
        0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF,
    };
    for (const auto opcode : kOpcodes) {
        const auto low = static_cast<std::uint8_t>(opcode & 0x0F);
        const bool scalar = low == 0x9 || low == 0xB || low == 0xD || low == 0xF;
        for (const bool wideW : {false, true}) {
            for (const bool vec256 : {false, true}) {
                const auto bytes = vexFmaReg(opcode, wideW, vec256, 11, 6, 14);
                const auto op = Codegen::DecodeFma(bytes.data(), bytes.size());
                require(op.has_value(), "valid FMA3 opcode was not decoded");
                require(op->DoublePrecision == wideW, "FMA3 precision mismatch");
                require(op->Scalar == scalar, "FMA3 scalar flag mismatch");
                require(op->Vector256 == (!scalar && vec256), "FMA3 vector width mismatch");
                require(op->Op1 == 11 && op->Op2 == 6 && op->Op3 == 14 && !op->Memory, "FMA3 register fields mismatch");
                const auto memBytes = vexFmaMemBase(opcode, wideW, vec256, 9, 13, 3);
                const auto memOp = Codegen::DecodeFma(memBytes.data(), memBytes.size());
                require(memOp && memOp->Memory && memOp->Op1 == 9 && memOp->Op2 == 13, "FMA3 memory form was not decoded");
            }
        }
    }
    const auto opNameCheck = [&](const std::uint8_t opcode, const bool wideW, const std::string& expected) {
        const auto bytes = vexFmaReg(opcode, wideW, false, 0, 1, 2);
        const auto op = Codegen::DecodeFma(bytes.data(), bytes.size());
        require(op && expected == op->Name, "unexpected FMA3 mnemonic: expected " + expected);
    };
    opNameCheck(0x96, false, "VFMADDSUB132PS");
    opNameCheck(0x97, true, "VFMSUBADD132PD");
    opNameCheck(0xA8, false, "VFMADD213PS");
    opNameCheck(0xA9, true, "VFMADD213SD");
    opNameCheck(0xBA, false, "VFMSUB231PS");
    opNameCheck(0xBD, false, "VFNMADD231SS");
    opNameCheck(0xBE, true, "VFNMSUB231PD");
    opNameCheck(0xBF, true, "VFNMSUB231SD");

    for (const Bytes& invalid : {
             Bytes{0xC4, 0xE2, 0x79, 0x90, 0x00},
             Bytes{0xC4, 0xE2, 0x79, 0xA0, 0xC0},
             Bytes{0xC4, 0xE2, 0x79, 0xB5, 0xC0},
             Bytes{0xC4, 0xE2, 0x78, 0xA8, 0xC0},
             Bytes{0xC4, 0xE2, 0x7A, 0xA8, 0xC0},
             Bytes{0xC4, 0xE2, 0x7B, 0xA8, 0xC0},
             Bytes{0xC4, 0xE1, 0x79, 0xA8, 0xC0},
             Bytes{0xC4, 0xE3, 0x79, 0xA8, 0xC0, 0x00},
             Bytes{0xC5, 0xF9, 0xA8, 0xC0},
             Bytes{0x66, 0xC4, 0xE2, 0x79, 0xA8, 0xC0},
             Bytes{0x90},
         }) {
        require(!Codegen::DecodeFma(invalid.data(), invalid.size()), "non-FMA3 instruction was decoded as FMA3");
    }

    auto trailing = vexFmaReg(0xA8, false, false, 0, 1, 2);
    trailing.push_back(0x90);
    bool threwTrailing = false;
    try { (void)Codegen::DecodeFma(trailing.data(), trailing.size()); }
    catch (const Codegen::CodegenException&) { threwTrailing = true; }
    require(threwTrailing, "trailing bytes on FMA3 instruction were accepted");

    const auto overflowDisp = vexFmaRspDisp(0xA8, false, true, 0, 1, 0x7FFFFF00);
    bool threwDisp = false;
    try {
        Codegen::StubBodyBuilder body;
        Codegen::EmitFma(body, *Codegen::DecodeFma(overflowDisp.data(), overflowDisp.size()));
    } catch (const Codegen::CodegenException&) {
        threwDisp = true;
    }
    require(threwDisp, "overflowing stack displacement was not rejected");
}

void matching() {
    const auto avx = Codegen::MakeAmd64OnlyInstructionMatcher(Codegen::Amd64OnlyTarget::Avx);
    const auto rosetta = Codegen::MakeAmd64OnlyInstructionMatcher(Codegen::Amd64OnlyTarget::Rosetta);
    const auto intel = Codegen::MakeAmd64OnlyInstructionMatcher(Codegen::Amd64OnlyTarget::Intel);
    const auto site = vexFmaReg(0xB8, true, true, 2, 3, 4);
    require(!rosetta->Match(site.data(), site.size()), "Rosetta target lowered FMA3");
    require(!intel->Match(site.data(), site.size()), "Intel target lowered FMA3");
    const auto match = avx->Match(site.data(), site.size());
    require(match && match->InstructionName == "VFMADD231PD" && match->Lowering == Codegen::Amd64OnlyLowering::Trampoline && match->RelocatableWindow, "AVX target did not match VFMADD231PD");
    Codegen::StubBodyBuilder body;
    avx->EmitOutOfLine(body, site);
    const auto finished = body.Finish();
    require(!finished.Bytes.empty(), "EmitOutOfLine produced empty FMA3 body");

    const Codegen::X64InstructionDecoder decoder;
    for (const auto& sample : {
             vexFmaReg(0xA8, false, true, 0, 1, 2),
             vexFmaReg(0xB8, true, true, 12, 13, 14),
             vexFmaMemBase(0xA9, false, false, 3, 4, 0),
             vexFmaMemBase(0xB9, true, false, 9, 10, 3),
         }) {
        Codegen::StubBodyBuilder stub;
        Codegen::EmitFma(stub, *Codegen::DecodeFma(sample.data(), sample.size()));
        const auto emitted = stub.Finish();
        std::size_t offset = 0;
        while (offset < emitted.ReturnBranchOffset) {
            const auto info = decoder.DecodeInstruction(emitted.Bytes.data() + offset, emitted.ReturnBranchOffset - offset);
            require(info.Length > 0 && offset + info.Length <= emitted.ReturnBranchOffset, "emitted FMA3 stub failed to decode");
            require(!avx->Match(emitted.Bytes.data() + offset, info.Length), "emitted FMA3 stub contains an instruction unsupported on AVX1");
            offset += info.Length;
        }
    }
}

#if LOWERING_EXECUTION_AVAILABLE

struct FmaMachine {
    std::array<Vector, 16> Ymm{};
    std::array<std::uint64_t, 16> GprIn{};
    std::array<std::uint64_t, 16> GprOut{};
    std::array<std::uint64_t, 16> RedZone{};
    std::uint64_t FlagsIn = LoweringExecution::kBaseFlags | LoweringExecution::kFlagMask;
    std::uint64_t FlagsOut = 0;
    std::uint32_t MxcsrIn = 0x1F80u;
    std::uint32_t SavedMxcsr = 0;
    Vector RipData{};
    Vector DeepStackData{};
};

constexpr std::int32_t kOffYmm = 0;
constexpr std::int32_t kOffGprIn = 512;
constexpr std::int32_t kOffGprOut = 640;
constexpr std::int32_t kOffRedZone = 768;
constexpr std::int32_t kOffFlagsIn = 896;
constexpr std::int32_t kOffFlagsOut = 904;
constexpr std::int32_t kOffMxcsrIn = 912;
constexpr std::int32_t kOffSavedMxcsr = 916;
constexpr std::int32_t kOffRipData = 920;

static_assert(offsetof(FmaMachine, Ymm) == kOffYmm);
static_assert(offsetof(FmaMachine, GprIn) == kOffGprIn);
static_assert(offsetof(FmaMachine, GprOut) == kOffGprOut);
static_assert(offsetof(FmaMachine, RedZone) == kOffRedZone);
static_assert(offsetof(FmaMachine, FlagsIn) == kOffFlagsIn);
static_assert(offsetof(FmaMachine, FlagsOut) == kOffFlagsOut);
static_assert(offsetof(FmaMachine, MxcsrIn) == kOffMxcsrIn);
static_assert(offsetof(FmaMachine, SavedMxcsr) == kOffSavedMxcsr);
static_assert(offsetof(FmaMachine, RipData) == kOffRipData);

class FmaRunner {
public:
    FmaRunner(const Codegen::LoweredBody& body, const bool hasRipData = false, const bool initDeepStack = false) {
        Bytes code;
        const auto pushReg = [&](const std::uint8_t reg) {
            if (reg >= 8) code.push_back(0x41);
            code.push_back(static_cast<std::uint8_t>(0x50 | (reg & 7)));
        };
        const auto popReg = [&](const std::uint8_t reg) {
            if (reg >= 8) code.push_back(0x41);
            code.push_back(static_cast<std::uint8_t>(0x58 | (reg & 7)));
        };
        const auto emitGprMemBase = [&](const std::uint8_t op, const std::uint8_t reg, const std::uint8_t base, const std::int32_t disp) {
            code.push_back(static_cast<std::uint8_t>(0x48 | ((reg & 8) >> 1) | ((base & 8) >> 3)));
            code.push_back(op);
            code.push_back(static_cast<std::uint8_t>(0x80 | ((reg & 7) << 3) | (base & 7)));
            if ((base & 7) == 4) code.push_back(0x24);
            append32(code, disp);
        };
        const auto emitYmmMemRdi = [&](const std::uint8_t op, const std::uint8_t reg, const std::int32_t disp) {
            code.insert(code.end(), {0xC4, static_cast<std::uint8_t>((reg & 8 ? 0x60 : 0xE0) | 1), 0x7E, op, static_cast<std::uint8_t>(0x87 | ((reg & 7) << 3))});
            append32(code, disp);
        };

        for (const auto r : {std::uint8_t{3}, std::uint8_t{5}, std::uint8_t{12}, std::uint8_t{13}, std::uint8_t{14}, std::uint8_t{15}})
            pushReg(r);
        code.insert(code.end(), {0x48, 0x8B, 0x4F, 0x08});
        code.insert(code.end(), {0x48, 0x8B, 0x3F});
        code.insert(code.end(), {0x48, 0x89, 0xE2});
        code.insert(code.end(), {0x48, 0x89, 0xCC});
        code.insert(code.end(), {0x52, 0x57});

        code.insert(code.end(), {0xC5, 0xF8, 0xAE, 0x9F});
        append32(code, kOffSavedMxcsr);
        code.insert(code.end(), {0xC5, 0xF8, 0xAE, 0x97});
        append32(code, kOffMxcsrIn);

        for (std::uint8_t reg = 0; reg < 16; ++reg)
            emitYmmMemRdi(0x6F, reg, kOffYmm + reg * 32);

        code.insert(code.end(), {0xFF, 0xB7});
        append32(code, kOffFlagsIn);
        code.push_back(0x9D);

        if (initDeepStack) {
            for (int q = 0; q < 4; ++q) {
                emitGprMemBase(0x8B, 0, 7, static_cast<std::int32_t>(offsetof(FmaMachine, DeepStackData)) + q * 8);
                emitGprMemBase(0x89, 0, 4, -200 + q * 8);
            }
        }
        for (int slot = 0; slot < 16; ++slot) {
            emitGprMemBase(0x8B, 0, 7, kOffRedZone + slot * 8);
            emitGprMemBase(0x89, 0, 4, -128 + slot * 8);
        }

        for (std::uint8_t reg = 0; reg < 16; ++reg) {
            if (reg == 4 || reg == 7) continue;
            emitGprMemBase(0x8B, reg, 7, kOffGprIn + reg * 8);
        }
        emitGprMemBase(0x8B, 7, 7, kOffGprIn + 7 * 8);

        const auto stubStart = code.size();
        const auto stubLen = body.ReturnBranchOffset;
        code.insert(code.end(), body.Bytes.begin(), body.Bytes.begin() + static_cast<std::ptrdiff_t>(stubLen));

        code.insert(code.end(), {0x48, 0x8D, 0xA4, 0x24});
        append32(code, -128);
        code.push_back(0x9C);
        code.push_back(0x57);
        emitGprMemBase(0x8B, 7, 4, 144);
        for (std::uint8_t reg = 0; reg < 16; ++reg) {
            if (reg == 4 || reg == 7) continue;
            emitGprMemBase(0x89, reg, 7, kOffGprOut + reg * 8);
        }
        code.push_back(0x58);
        emitGprMemBase(0x89, 0, 7, kOffGprOut + 7 * 8);
        code.push_back(0x58);
        emitGprMemBase(0x89, 0, 7, kOffFlagsOut);
        code.insert(code.end(), {0x48, 0x8D, 0xA4, 0x24});
        append32(code, 128);

        code.insert(code.end(), {0xC5, 0xF8, 0xAE, 0x97});
        append32(code, kOffSavedMxcsr);

        for (std::uint8_t reg = 0; reg < 16; ++reg)
            emitYmmMemRdi(0x7F, reg, kOffYmm + reg * 32);

        for (int slot = 0; slot < 16; ++slot) {
            emitGprMemBase(0x8B, 0, 4, -128 + slot * 8);
            emitGprMemBase(0x89, 0, 7, kOffRedZone + slot * 8);
        }

        code.insert(code.end(), {0x48, 0x8B, 0x64, 0x24, 0x08, 0xC5, 0xF8, 0x77});
        for (const auto r : {std::uint8_t{15}, std::uint8_t{14}, std::uint8_t{13}, std::uint8_t{12}, std::uint8_t{5}, std::uint8_t{3}})
            popReg(r);
        code.push_back(0xC3);

        while ((code.size() & 31u) != 0) code.push_back(0x90);
        const auto ripDataOffset = code.size();
        if (hasRipData) code.insert(code.end(), 32, 0);

        for (const auto& reloc : body.Relocations) {
            const auto patchPos = stubStart + reloc.DisplacementOffset;
            const auto targetPos = ripDataOffset;
            const auto disp = static_cast<std::int64_t>(targetPos) - static_cast<std::int64_t>(patchPos + 4);
            const auto u32 = static_cast<std::uint32_t>(static_cast<std::int32_t>(disp));
            for (unsigned i = 0; i < 4; ++i)
                code[patchPos + i] = static_cast<std::uint8_t>(u32 >> (i * 8));
        }

        _entry = LoweringExecution::AppendHostEntry(code);
        _ripDataOffset = ripDataOffset;
        _hasRipData = hasRipData;
        if (code.size() > kPage) throw std::runtime_error("FMA runner page overflow");
        _page = LoweringExecution::AllocatePages(kPage);
        if (!_page) throw std::runtime_error("cannot allocate FMA runner page");
        std::memcpy(_page, code.data(), code.size());
        if (!hasRipData && !LoweringExecution::ProtectPages(_page, kPage, true, false)) {
            LoweringExecution::FreePages(_page, kPage);
            throw std::runtime_error("cannot mark FMA runner page executable");
        }
    }

    ~FmaRunner() { LoweringExecution::FreePages(_page, kPage); }
    FmaRunner(const FmaRunner&) = delete;
    FmaRunner& operator=(const FmaRunner&) = delete;

    void Run(FmaMachine& machine) {
        if (_hasRipData) {
            require(LoweringExecution::ProtectPages(_page, kPage, false, true), "cannot unprotect RIP data page");
            std::memcpy(_page + _ripDataOffset, machine.RipData.data(), 32);
            require(LoweringExecution::ProtectPages(_page, kPage, true, false), "cannot re-protect RIP data page");
        }
        alignas(64) std::array<std::uint8_t, 8192> stack{};
        struct {
            FmaMachine* Target;
            std::uint8_t* Top;
        } frame{&machine, stack.data() + stack.size()};
        LoweringExecution::Call(_page, _entry, &frame);
    }

private:
    static constexpr std::size_t kPage = 16384;
    std::uint8_t* _page = nullptr;
    std::size_t _entry = 0;
    std::size_t _ripDataOffset = 0;
    bool _hasRipData = false;
};

Codegen::LoweredBody loweredBody(const Bytes& instruction) {
    Codegen::StubBodyBuilder body;
    const auto op = Codegen::DecodeFma(instruction.data(), instruction.size());
    require(op.has_value(), "failed to decode FMA3 instruction");
    Codegen::EmitFma(body, *op);
    return body.Finish();
}

Codegen::LoweredBody rawInstructionBody(const Bytes& instruction) {
    Codegen::StubBodyBuilder body;
    body.Raw(instruction);
    return body.Finish();
}

FmaMachine initialMachine() {
    FmaMachine m{};
    for (std::size_t r = 0; r < 16; ++r) {
        m.GprIn[r] = 0x1122334455667700ull + r * 0x0101010101010101ull;
        for (std::size_t q = 0; q < 4; ++q)
            m.Ymm[r][q] = 0x3FF0000000000000ull ^ (r * 0x0001000200030004ull) ^ (q * 0x0010002000300040ull);
    }
    for (auto& slot : m.RedZone) slot = LoweringExecution::kSentinel;
    return m;
}

void verifyState(const FmaMachine& before, const FmaMachine& after, const std::uint8_t dst, const Vector& wanted, const std::string& label) {
    for (std::uint8_t r = 0; r < 16; ++r) {
        const auto& exp = r == dst ? wanted : before.Ymm[r];
        if (after.Ymm[r] != exp)
            throw std::runtime_error(label + ": ymm" + std::to_string(r) + " mismatch");
    }
    for (std::uint8_t r = 0; r < 16; ++r) {
        if (r == 4) continue;
        if (after.GprOut[r] != before.GprIn[r])
            throw std::runtime_error(label + ": gpr" + std::to_string(r) + " modified");
    }
    if ((after.FlagsOut & LoweringExecution::kFlagMask) != (before.FlagsIn & LoweringExecution::kFlagMask))
        throw std::runtime_error(label + ": RFLAGS modified");
    for (const auto slot : after.RedZone)
        require(slot == LoweringExecution::kSentinel, label + ": red zone overwritten");
}

void execution() {
    const Bytes avxProbe{0xC5, 0xFC, 0x57, 0xC0};
    if (!LoweringExecution::HostExecutes(avxProbe)) {
        std::cout << "fma execution checks skipped on a host without AVX\n";
        return;
    }
    const bool hasHwFma = LoweringExecution::HostExecutes(vexFmaReg(0xA9, false, false, 0, 0, 0));

    constexpr std::array<std::uint8_t, 30> kAllOpcodes{
        0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF,
        0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF,
    };

    const auto packF32 = [](std::array<std::uint32_t, 8> lanes) {
        Vector v{};
        std::memcpy(v.data(), lanes.data(), 32);
        return v;
    };
    const auto packF64 = [](std::array<std::uint64_t, 4> lanes) {
        Vector v{};
        std::memcpy(v.data(), lanes.data(), 32);
        return v;
    };

    const Vector sampleF32A = packF32({0x3FC00000u, 0xC0200000u, 0x00800000u, 0x7F7FFFFFu, 0xBF800000u, 0x00000001u, 0x3F800001u, 0x80000000u});
    const Vector sampleF32B = packF32({0x40400000u, 0x3F000000u, 0x3F7FFFFFu, 0x40000000u, 0x40400000u, 0x3F800000u, 0x3F800001u, 0x3F800000u});
    const Vector sampleF32C = packF32({0x40A00000u, 0x3FA00000u, 0x00000000u, 0xFF7FFFFFu, 0x40400000u, 0x00000001u, 0xBF800002u, 0x00000000u});

    const Vector sampleF64A = packF64({0x3FF8000000000000ull, 0xC004000000000000ull, 0x7FEFFFFFFFFFFFFFull, 0x3FF0000000000001ull});
    const Vector sampleF64B = packF64({0x4008000000000000ull, 0x3FE0000000000000ull, 0x4000000000000000ull, 0x3FF0000000000001ull});
    const Vector sampleF64C = packF64({0x4014000000000000ull, 0x3FF4000000000000ull, 0xFFEFFFFFFFFFFFFFull, 0xBFF0000000000002ull});

    for (const auto opcode : kAllOpcodes) {
        for (const bool wideW : {false, true}) {
            for (const bool vec256 : {false, true}) {
                const auto regSite = vexFmaReg(opcode, wideW, vec256, 2, 11, 14);
                const auto op = *Codegen::DecodeFma(regSite.data(), regSite.size());
                FmaRunner runner(loweredBody(regSite));
                std::unique_ptr<FmaRunner> hwRunner;
                if (hasHwFma) hwRunner = std::make_unique<FmaRunner>(rawInstructionBody(regSite));

                auto m = initialMachine();
                m.Ymm[2] = wideW ? sampleF64A : sampleF32A;
                m.Ymm[11] = wideW ? sampleF64B : sampleF32B;
                m.Ymm[14] = wideW ? sampleF64C : sampleF32C;
                const auto start = m;
                runner.Run(m);
                const auto wanted = sdmFmaVector(op, start.Ymm[2], start.Ymm[11], start.Ymm[14], start.MxcsrIn);
                verifyState(start, m, 2, wanted, std::string(op.Name) + (vec256 ? " L=1" : " L=0"));
                if (hwRunner) {
                    auto hw = start;
                    hwRunner->Run(hw);
                    require(m.Ymm[2] == hw.Ymm[2], std::string(op.Name) + " differed from hardware FMA3");
                }
            }
        }
    }

    for (const auto& [r1, r2, r3] : std::array<std::array<std::uint8_t, 3>, 5>{{
             {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {15, 15, 15}, {8, 9, 10},
         }}) {
        for (const auto opcode : {std::uint8_t{0x98}, std::uint8_t{0xA8}, std::uint8_t{0xB8}, std::uint8_t{0xA9}}) {
            const auto site = vexFmaReg(opcode, false, true, r1, r2, r3);
            const auto op = *Codegen::DecodeFma(site.data(), site.size());
            FmaRunner runner(loweredBody(site));
            auto m = initialMachine();
            m.Ymm[r1] = sampleF32A;
            if (r2 != r1) m.Ymm[r2] = sampleF32B;
            if (r3 != r1 && r3 != r2) m.Ymm[r3] = sampleF32C;
            const auto start = m;
            runner.Run(m);
            const auto wanted = sdmFmaVector(op, start.Ymm[r1], start.Ymm[r2], start.Ymm[r3], start.MxcsrIn);
            verifyState(start, m, r1, wanted, std::string(op.Name) + " alias");
        }
    }

    const std::array<std::uint32_t, 16> kMxcsrModes{
        0x1F80u, 0x3F80u, 0x5F80u, 0x7F80u,
        0x1FC0u, 0x3FC0u, 0x5FC0u, 0x7FC0u,
        0x9F80u, 0xBF80u, 0xDF80u, 0xFF80u,
        0x9FC0u, 0xBFC0u, 0xDFC0u, 0xFFC0u,
    };

    {
        struct Case32 { std::uint32_t a, b, c; };
        std::vector<Case32> cases32{
            {0x00000000u, 0x3F800000u, 0x00000000u},
            {0x80000000u, 0x3F800000u, 0x00000000u},
            {0x00000000u, 0x3F800000u, 0x80000000u},
            {0x80000000u, 0x3F800000u, 0x80000000u},
            {0x3F800000u, 0x3F800000u, 0xBF800000u},
            {0x00000000u, 0x7F800000u, 0x00000000u},
            {0x00000000u, 0x7F800000u, 0x7FC00005u},
            {0x7F800000u, 0x3F800000u, 0xFF800000u},
            {0x7FC00001u, 0x7FC00002u, 0x7FC00003u},
            {0x3F800000u, 0x7F800002u, 0x7FC00003u},
            {0x3F800000u, 0x3F800000u, 0xFFC00007u},
            {0x7F7FFFFFu, 0x40000000u, 0xFF7FFFFFu},
            {0xFF7FFFFFu, 0x40000000u, 0x7F7FFFFFu},
            {0x7F7FFFFFu, 0x40000000u, 0x00000000u},
            {0xFF7FFFFFu, 0x40000000u, 0x00000000u},
            {0x00000001u, 0x7F800000u, 0x00000000u},
            {0x00800000u, 0x3F7FFFFFu, 0x00000000u},
            {0x80800000u, 0x33000000u, 0x00800000u},
            {0x80800000u, 0x33000001u, 0x00800000u},
            {0x80800000u, 0x33800000u, 0x00800000u},
            {0x80800000u, 0x337FFFFFu, 0x00800000u},
            {0x1FFFFFFEu, 0x20000001u, 0x00000000u},
            {0x00000001u, 0x00000001u, 0x00000000u},
            {0x00000001u, 0x00000001u, 0x00800000u},
            {0x3F800001u, 0x3F800001u, 0xBF800002u},
            {0x3F800001u, 0x3F800001u, 0x33800000u},
            {0x3F800001u, 0x3F7FFFFEu, 0x4B800001u},
            {0x3F800000u, 0x3F800000u, 0x00000001u},
            {0x3F800000u, 0x3F800000u, 0x80000001u},
            {0x00000000u, 0x3F800000u, 0x00000001u},
            {0x00000000u, 0x3F800000u, 0x80000001u},
        };
        std::uint64_t rng = 0x9E3779B97F4A7C15ull;
        const auto next = [&]() {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            return rng * 0x2545F4914F6CDD1Dull;
        };
        for (int i = 0; i < 160; ++i) {
            if ((i & 1) == 0) {
                cases32.push_back({static_cast<std::uint32_t>(next()), static_cast<std::uint32_t>(next()), static_cast<std::uint32_t>(next())});
            } else {
                const int ea = static_cast<int>(next() % 240) + 5;
                const int eb = static_cast<int>(next() % 240) + 5;
                int ec = ea + eb - 127 + static_cast<int>(next() % 7) - 3;
                if (ec < 0) ec = 0;
                if (ec > 254) ec = 254;
                cases32.push_back({
                    static_cast<std::uint32_t>(((next() & 1) << 31) | (ea << 23) | (next() & 0x7FFFFFu)),
                    static_cast<std::uint32_t>(((next() & 1) << 31) | (eb << 23) | (next() & 0x7FFFFFu)),
                    static_cast<std::uint32_t>(((next() & 1) << 31) | (ec << 23) | (next() & 0x7FFFFFu)),
                });
            }
        }
        while (cases32.size() % 8 != 0) cases32.push_back(cases32.front());

        for (const auto opcode : {std::uint8_t{0xA8}, std::uint8_t{0xAA}, std::uint8_t{0xAC}, std::uint8_t{0xAE}}) {
            const auto site = vexFmaReg(opcode, false, true, 1, 0, 2);
            const auto op = *Codegen::DecodeFma(site.data(), site.size());
            FmaRunner runner(loweredBody(site));
            std::unique_ptr<FmaRunner> hwRunner;
            if (hasHwFma) hwRunner = std::make_unique<FmaRunner>(rawInstructionBody(site));
            for (const auto mxcsr : kMxcsrModes) {
                for (std::size_t base = 0; base < cases32.size(); base += 8) {
                    std::array<std::uint32_t, 8> va{}, vb{}, vc{};
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        va[lane] = cases32[base + lane].a;
                        vb[lane] = cases32[base + lane].b;
                        vc[lane] = cases32[base + lane].c;
                    }
                    auto m = initialMachine();
                    m.MxcsrIn = mxcsr;
                    m.Ymm[0] = packF32(va);
                    m.Ymm[1] = packF32(vb);
                    m.Ymm[2] = packF32(vc);
                    const auto start = m;
                    runner.Run(m);
                    const auto wanted = sdmFmaVector(op, start.Ymm[1], start.Ymm[0], start.Ymm[2], mxcsr);
                    verifyState(start, m, 1, wanted, std::string(op.Name) + " f32 stress");
                    if (hwRunner) {
                        auto hw = start;
                        hwRunner->Run(hw);
                        require(m.Ymm[1] == hw.Ymm[1], std::string(op.Name) + " f32 stress differed from hardware");
                    }
                }
            }
        }
    }

    {
        struct Case64 { std::uint64_t a, b, c; };
        std::vector<Case64> cases64{
            {0x0000000000000000ull, 0x3FF0000000000000ull, 0x0000000000000000ull},
            {0x8000000000000000ull, 0x3FF0000000000000ull, 0x0000000000000000ull},
            {0x0000000000000000ull, 0x3FF0000000000000ull, 0x8000000000000000ull},
            {0x8000000000000000ull, 0x3FF0000000000000ull, 0x8000000000000000ull},
            {0x3FF0000000000000ull, 0x3FF0000000000000ull, 0xBFF0000000000000ull},
            {0x0000000000000000ull, 0x7FF0000000000000ull, 0x0000000000000000ull},
            {0x0000000000000000ull, 0x7FF0000000000000ull, 0x7FF8000000000005ull},
            {0x7FF0000000000000ull, 0x3FF0000000000000ull, 0xFFF0000000000000ull},
            {0x7FF8000000000001ull, 0x7FF8000000000002ull, 0x7FF8000000000003ull},
            {0x3FF0000000000000ull, 0x7FF0000000000002ull, 0x7FF8000000000003ull},
            {0x7FEFFFFFFFFFFFFFull, 0x4000000000000000ull, 0xFFEFFFFFFFFFFFFFull},
            {0xFFEFFFFFFFFFFFFFull, 0x4000000000000000ull, 0x7FEFFFFFFFFFFFFFull},
            {0x7FEFFFFFFFFFFFFFull, 0x4000000000000000ull, 0x0000000000000000ull},
            {0xFFEFFFFFFFFFFFFFull, 0x4000000000000000ull, 0x0000000000000000ull},
            {0x0000000000000001ull, 0x7FF0000000000000ull, 0x0000000000000000ull},
            {0x8010000000000000ull, 0x3C90000000000000ull, 0x0010000000000000ull},
            {0x8010000000000000ull, 0x3C90000000000001ull, 0x0010000000000000ull},
            {0x8010000000000000ull, 0x3CA0000000000000ull, 0x0010000000000000ull},
            {0x0000000000000001ull, 0x0000000000000001ull, 0x0000000000000000ull},
            {0x0000000000000001ull, 0x0000000000000001ull, 0x0010000000000000ull},
            {0x3FF0000000000001ull, 0x3FF0000000000001ull, 0xBFF0000000000002ull},
            {0x3FF0000000000001ull, 0x3FEFFFFFFFFFFFFEull, 0x4340000000000001ull},
            {0x3FF0000000000000ull, 0x3FF0000000000000ull, 0x0000000000000001ull},
            {0x3FF0000000000000ull, 0x3FF0000000000000ull, 0x8000000000000001ull},
            {0x7FEFFFFFFFFFFFFFull, 0x3FF0000000000001ull, 0xFFEFFFFFFFFFFFFFull},
        };
        std::uint64_t rng = 0xD1B54A32D192ED03ull;
        const auto next = [&]() {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            return rng * 0x2545F4914F6CDD1Dull;
        };
        for (int i = 0; i < 120; ++i) {
            if ((i & 3) == 0) {
                cases64.push_back({next(), next(), next()});
            } else if ((i & 3) == 1) {
                const int ea = static_cast<int>(next() % 2000) + 10;
                const int eb = static_cast<int>(next() % 2000) + 10;
                int ec = ea + eb - 1023 + static_cast<int>(next() % 7) - 3;
                if (ec < 0) ec = 0;
                if (ec > 2046) ec = 2046;
                cases64.push_back({
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(ea) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(eb) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(ec) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                });
            } else if ((i & 3) == 2) {
                const int ea = static_cast<int>(next() % 1400) + 300;
                const int ec = ea - static_cast<int>(next() % 220);
                cases64.push_back({
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(ea) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(1023) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(ec) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                });
            } else {
                const int ea = static_cast<int>(next() % 1030);
                int eb = 1023 - ea + static_cast<int>(next() % 5) - 2;
                if (eb < 0) eb = 0;
                if (eb > 2046) eb = 2046;
                const int ec = static_cast<int>(next() % 4);
                cases64.push_back({
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(ea) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(eb) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                    ((next() & 1) << 63) | (static_cast<std::uint64_t>(ec) << 52) | (next() & 0x000FFFFFFFFFFFFFull),
                });
            }
        }
        while (cases64.size() % 4 != 0) cases64.push_back(cases64.front());

        for (const auto opcode : {std::uint8_t{0xA8}, std::uint8_t{0xAA}, std::uint8_t{0xAC}, std::uint8_t{0xAE}}) {
            const auto site = vexFmaReg(opcode, true, true, 1, 0, 2);
            const auto op = *Codegen::DecodeFma(site.data(), site.size());
            FmaRunner runner(loweredBody(site));
            std::unique_ptr<FmaRunner> hwRunner;
            if (hasHwFma) hwRunner = std::make_unique<FmaRunner>(rawInstructionBody(site));
            for (const auto mxcsr : kMxcsrModes) {
                for (std::size_t base = 0; base < cases64.size(); base += 4) {
                    std::array<std::uint64_t, 4> va{}, vb{}, vc{};
                    for (std::size_t lane = 0; lane < 4; ++lane) {
                        va[lane] = cases64[base + lane].a;
                        vb[lane] = cases64[base + lane].b;
                        vc[lane] = cases64[base + lane].c;
                    }
                    auto m = initialMachine();
                    m.MxcsrIn = mxcsr;
                    m.Ymm[0] = packF64(va);
                    m.Ymm[1] = packF64(vb);
                    m.Ymm[2] = packF64(vc);
                    const auto start = m;
                    runner.Run(m);
                    const auto wanted = sdmFmaVector(op, start.Ymm[1], start.Ymm[0], start.Ymm[2], mxcsr);
                    verifyState(start, m, 1, wanted, std::string(op.Name) + " f64 stress");
                    if (hwRunner) {
                        auto hw = start;
                        hwRunner->Run(hw);
                        require(m.Ymm[1] == hw.Ymm[1], std::string(op.Name) + " f64 stress differed from hardware");
                    }
                }
            }
        }
    }

    {
        const auto ripSite = vexFmaRip(0xB8, true, true, 3, 5, 0);
        const auto op = *Codegen::DecodeFma(ripSite.data(), ripSite.size());
        FmaRunner runner(loweredBody(ripSite), true, false);
        auto m = initialMachine();
        m.Ymm[3] = sampleF64C;
        m.Ymm[5] = sampleF64A;
        m.RipData = sampleF64B;
        const auto start = m;
        runner.Run(m);
        verifyState(start, m, 3, sdmFmaVector(op, start.Ymm[3], start.Ymm[5], start.RipData, start.MxcsrIn), "RIP-relative VFMADD231PD");
    }

    {
        alignas(32) std::array<std::uint8_t, 128> buf{};
        std::memcpy(buf.data() + 32, sampleF32C.data(), 32);
        const auto sibSite = vexFmaSib(0xA8, false, true, 4, 6, 3, 1, 2, 16);
        const auto op = *Codegen::DecodeFma(sibSite.data(), sibSite.size());
        FmaRunner runner(loweredBody(sibSite));
        auto m = initialMachine();
        m.GprIn[3] = reinterpret_cast<std::uintptr_t>(buf.data());
        m.GprIn[1] = 4;
        m.Ymm[4] = sampleF32B;
        m.Ymm[6] = sampleF32A;
        const auto start = m;
        runner.Run(m);
        verifyState(start, m, 4, sdmFmaVector(op, start.Ymm[4], start.Ymm[6], sampleF32C, start.MxcsrIn), "SIB VFMADD213PS");
    }

    {
        const auto rspSite = vexFmaRspDisp(0x98, true, true, 0, 1, -200);
        const auto op = *Codegen::DecodeFma(rspSite.data(), rspSite.size());
        FmaRunner runner(loweredBody(rspSite), false, true);
        auto m = initialMachine();
        m.Ymm[0] = sampleF64A;
        m.Ymm[1] = sampleF64C;
        m.DeepStackData = sampleF64B;
        const auto start = m;
        runner.Run(m);
        verifyState(start, m, 0, sdmFmaVector(op, start.Ymm[0], start.Ymm[1], start.DeepStackData, start.MxcsrIn), "deep RSP VFMADD132PD");
    }

    {
        const auto pageSize = LoweringExecution::PageSize();
        auto* area = LoweringExecution::AllocatePages(pageSize * 2);
        require(area != nullptr, "cannot allocate guard pages");
        require(LoweringExecution::ProtectPages(area + pageSize, pageSize, false, false), "cannot protect guard page");

        const auto checkPageEnd = [&](const std::uint8_t opcode, const bool wideW, const bool vec256, const std::size_t width) {
            auto* operand = area + pageSize - width;
            const Vector src = wideW ? sampleF64B : sampleF32B;
            std::memcpy(operand, src.data(), width);
            const auto site = vexFmaMemBase(opcode, wideW, vec256, 7, 8, 0);
            const auto op = *Codegen::DecodeFma(site.data(), site.size());
            FmaRunner runner(loweredBody(site));
            auto m = initialMachine();
            m.GprIn[0] = reinterpret_cast<std::uintptr_t>(operand);
            m.Ymm[7] = wideW ? sampleF64A : sampleF32A;
            m.Ymm[8] = wideW ? sampleF64C : sampleF32C;
            const auto start = m;
            runner.Run(m);
            verifyState(start, m, 7, sdmFmaVector(op, start.Ymm[7], start.Ymm[8], src, start.MxcsrIn), std::string(op.Name) + " page-end");
        };
        checkPageEnd(0xA9, false, false, 4);
        checkPageEnd(0xA9, true, false, 8);
        checkPageEnd(0xA8, false, false, 16);
        checkPageEnd(0xA8, true, false, 16);
        checkPageEnd(0xA8, false, true, 32);
        checkPageEnd(0xA8, true, true, 32);
        LoweringExecution::FreePages(area, pageSize * 2);
    }

    std::cout << (hasHwFma ? "fma execution verified against SDM and hardware\n" : "fma execution verified against SDM\n");
}

#else

void execution() { std::cout << "fma execution checks require an x86-64 Linux, macOS or Windows build\n"; }

#endif

}

int main() {
    try {
        decoding();
        matching();
        execution();
        std::cout << "fma lowering checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
