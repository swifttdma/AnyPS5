#ifndef CODEGEN_X86_FMALOWERING_HPP
#define CODEGEN_X86_FMALOWERING_HPP

#include <codegen/x86/StubBodyBuilder.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace Codegen {

enum class FmaOperation : std::uint8_t {
    Madd,
    Msub,
    Nmadd,
    Nmsub,
    MaddSub,
    MsubAdd,
};

enum class FmaOrder : std::uint8_t {
    Order132,
    Order213,
    Order231,
};

struct FmaOperands {
    const char* Name;
    FmaOperation Operation;
    FmaOrder Order;
    bool DoublePrecision;
    bool Scalar;
    bool Vector256;
    std::uint8_t Op1;
    std::uint8_t Op2;
    std::uint8_t Op3;
    std::optional<MemoryOperand> Memory;
};

[[nodiscard]] std::optional<FmaOperands> DecodeFma(const std::uint8_t* data, std::size_t length);

void EmitFma(StubBodyBuilder& body, const FmaOperands& operands);

}

#endif
