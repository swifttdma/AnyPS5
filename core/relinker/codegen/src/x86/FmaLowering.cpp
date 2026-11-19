#include <codegen/x86/FmaLowering.hpp>
#include <codegen/x86/ScalarLowering.hpp>
#include <codegen/x86/VexEncoding.hpp>
#include <codegen/CodegenException.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace Codegen {

using namespace ScalarLowering;
using namespace VexEncoding;

namespace {

constexpr std::uint8_t kPushFlags = 0x9C;
constexpr std::uint8_t kPopFlags = 0x9D;

constexpr std::int32_t kSlotMxcsr = 8;
constexpr std::int32_t kSlotGprBase = 16;
constexpr std::int32_t kLocal0 = 136;
constexpr std::int32_t kLocal1 = 144;
constexpr std::int32_t kBufOp1 = 192;
constexpr std::int32_t kBufOp2 = 224;
constexpr std::int32_t kBufOp3 = 256;
constexpr std::int32_t kSaveYmm = 288;
constexpr std::int32_t kSpillBytes = 320;
constexpr std::int32_t kDefaultFrame = 128 + kSpillBytes;

constexpr Form kMoveUnaligned128Store{Map0F, PrefixF3, false, false, 0x7F};
constexpr Form kLoadScalarSingle{Map0F, PrefixF3, false, false, 0x10};
constexpr Form kLoadScalarDouble{Map0F, PrefixF2, false, false, 0x10};
constexpr Form kStoreMxcsr{Map0F, 0, false, false, 0xAE};

constexpr std::array<std::uint8_t, 15> kSavedGprs{0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

constexpr std::uint8_t kRax = 0;
constexpr std::uint8_t kRcx = 1;
constexpr std::uint8_t kRdx = 2;
constexpr std::uint8_t kRbx = 3;
constexpr std::uint8_t kRbp = 5;
constexpr std::uint8_t kRsi = 6;
constexpr std::uint8_t kRdi = 7;
constexpr std::uint8_t kR8 = 8;
constexpr std::uint8_t kR9 = 9;
constexpr std::uint8_t kR10 = 10;
constexpr std::uint8_t kR11 = 11;
constexpr std::uint8_t kR12 = 12;
constexpr std::uint8_t kR13 = 13;
constexpr std::uint8_t kR14 = 14;
constexpr std::uint8_t kR15 = 15;

constexpr std::uint8_t kCcB = 0x2;
constexpr std::uint8_t kCcAe = 0x3;
constexpr std::uint8_t kCcE = 0x4;
constexpr std::uint8_t kCcNe = 0x5;
constexpr std::uint8_t kCcBe = 0x6;
constexpr std::uint8_t kCcL = 0xC;
constexpr std::uint8_t kCcGe = 0xD;
constexpr std::uint8_t kCcLe = 0xE;
constexpr std::uint8_t kCcG = 0xF;

struct Label {
    std::optional<std::size_t> Pos;
    std::vector<std::size_t> Fixups;
};

class Emitter {
public:
    std::vector<std::uint8_t> Code;

    void Bind(Label& label) {
        if (label.Pos)
            throw CodegenException("FMA label bound twice");
        label.Pos = Code.size();
        for (const auto fixup : label.Fixups) {
            const auto rel = static_cast<std::int64_t>(*label.Pos) - static_cast<std::int64_t>(fixup + 4);
            if (rel < std::numeric_limits<std::int32_t>::min() || rel > std::numeric_limits<std::int32_t>::max())
                throw CodegenException("FMA branch displacement overflow");
            const auto value = static_cast<std::uint32_t>(static_cast<std::int32_t>(rel));
            for (std::size_t i = 0; i < 4; ++i)
                Code[fixup + i] = static_cast<std::uint8_t>(value >> (i * 8));
        }
        label.Fixups.clear();
    }

    void Jmp(Label& label) {
        Code.push_back(0xE9);
        _emitRel32(label);
    }

    void Jcc(const std::uint8_t cc, Label& label) {
        Code.push_back(0x0F);
        Code.push_back(static_cast<std::uint8_t>(0x80 | cc));
        _emitRel32(label);
    }

    void Mov64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x8B, true, dst, src); }
    void Mov32(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x8B, false, dst, src); }
    void Add64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x03, true, dst, src); }
    void Adc64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x13, true, dst, src); }
    void Sub64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x2B, true, dst, src); }
    void Sbb64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x1B, true, dst, src); }
    void And64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x23, true, dst, src); }
    void Or64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x0B, true, dst, src); }
    void Xor64(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x33, true, dst, src); }
    void Cmp64(const std::uint8_t lhs, const std::uint8_t rhs) { _opRr(0x3B, true, lhs, rhs); }
    void Test64(const std::uint8_t lhs, const std::uint8_t rhs) { _opRr(0x85, true, rhs, lhs); }

    void Add32(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x03, false, dst, src); }
    void Sub32(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x2B, false, dst, src); }
    void And32(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x23, false, dst, src); }
    void Or32(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x0B, false, dst, src); }
    void Xor32(const std::uint8_t dst, const std::uint8_t src) { _opRr(0x33, false, dst, src); }
    void Cmp32(const std::uint8_t lhs, const std::uint8_t rhs) { _opRr(0x3B, false, lhs, rhs); }
    void Test32(const std::uint8_t lhs, const std::uint8_t rhs) { _opRr(0x85, false, rhs, lhs); }

    void MovImm32(const std::uint8_t dst, const std::uint32_t imm) {
        if (dst >= 8) Code.push_back(0x41);
        Code.push_back(static_cast<std::uint8_t>(0xB8 | (dst & 7)));
        _appendU32(imm);
    }

    void MovImm64(const std::uint8_t dst, const std::uint64_t imm) {
        Code.push_back(static_cast<std::uint8_t>(0x48 | ((dst & 8) >> 3)));
        Code.push_back(static_cast<std::uint8_t>(0xB8 | (dst & 7)));
        for (unsigned shift = 0; shift < 64; shift += 8)
            Code.push_back(static_cast<std::uint8_t>(imm >> shift));
    }

    void AddImm32(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(false, 0, dst, static_cast<std::uint32_t>(imm)); }
    void OrImm32(const std::uint8_t dst, const std::uint32_t imm) { _aluImm32(false, 1, dst, imm); }
    void AndImm32(const std::uint8_t dst, const std::uint32_t imm) { _aluImm32(false, 4, dst, imm); }
    void SubImm32(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(false, 5, dst, static_cast<std::uint32_t>(imm)); }
    void XorImm32(const std::uint8_t dst, const std::uint32_t imm) { _aluImm32(false, 6, dst, imm); }
    void CmpImm32(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(false, 7, dst, static_cast<std::uint32_t>(imm)); }

    void AddImm64(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(true, 0, dst, static_cast<std::uint32_t>(imm)); }
    void OrImm64(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(true, 1, dst, static_cast<std::uint32_t>(imm)); }
    void AndImm64(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(true, 4, dst, static_cast<std::uint32_t>(imm)); }
    void SubImm64(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(true, 5, dst, static_cast<std::uint32_t>(imm)); }
    void CmpImm64(const std::uint8_t dst, const std::int32_t imm) { _aluImm32(true, 7, dst, static_cast<std::uint32_t>(imm)); }

    void TestImm32(const std::uint8_t dst, const std::uint32_t imm) {
        if (dst >= 8) Code.push_back(0x41);
        Code.push_back(0xF7);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | (dst & 7)));
        _appendU32(imm);
    }

    void TestImm64(const std::uint8_t dst, const std::int32_t imm) {
        Code.push_back(static_cast<std::uint8_t>(0x48 | ((dst & 8) >> 3)));
        Code.push_back(0xF7);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | (dst & 7)));
        _appendU32(static_cast<std::uint32_t>(imm));
    }

    void ShlImm32(const std::uint8_t dst, const std::uint8_t count) { _shiftImm(false, 4, dst, count); }
    void ShrImm32(const std::uint8_t dst, const std::uint8_t count) { _shiftImm(false, 5, dst, count); }
    void ShlImm64(const std::uint8_t dst, const std::uint8_t count) { _shiftImm(true, 4, dst, count); }
    void ShrImm64(const std::uint8_t dst, const std::uint8_t count) { _shiftImm(true, 5, dst, count); }

    void ShlCl32(const std::uint8_t dst) { _shiftCl(false, 4, dst); }
    void ShlCl64(const std::uint8_t dst) { _shiftCl(true, 4, dst); }
    void ShrCl64(const std::uint8_t dst) { _shiftCl(true, 5, dst); }

    void ShldImm64(const std::uint8_t dst, const std::uint8_t src, const std::uint8_t count) {
        _emitRex(true, src, 0, dst);
        Code.push_back(0x0F);
        Code.push_back(0xA4);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((src & 7) << 3) | (dst & 7)));
        Code.push_back(count);
    }

    void ShldCl64(const std::uint8_t dst, const std::uint8_t src) {
        _emitRex(true, src, 0, dst);
        Code.push_back(0x0F);
        Code.push_back(0xA5);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((src & 7) << 3) | (dst & 7)));
    }

    void ShrdCl64(const std::uint8_t dst, const std::uint8_t src) {
        _emitRex(true, src, 0, dst);
        Code.push_back(0x0F);
        Code.push_back(0xAD);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((src & 7) << 3) | (dst & 7)));
    }

    void Imul64(const std::uint8_t dst, const std::uint8_t src) {
        _emitRex(true, dst, 0, src);
        Code.push_back(0x0F);
        Code.push_back(0xAF);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((dst & 7) << 3) | (src & 7)));
    }

    void Mul64(const std::uint8_t src) {
        _emitRex(true, 0, 0, src);
        Code.push_back(0xF7);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | (4 << 3) | (src & 7)));
    }

    void Bsr32(const std::uint8_t dst, const std::uint8_t src) {
        _emitRexOpt(false, dst, 0, src);
        Code.push_back(0x0F);
        Code.push_back(0xBD);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((dst & 7) << 3) | (src & 7)));
    }

    void Bsr64(const std::uint8_t dst, const std::uint8_t src) {
        _emitRex(true, dst, 0, src);
        Code.push_back(0x0F);
        Code.push_back(0xBD);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((dst & 7) << 3) | (src & 7)));
    }

    void Setcc(const std::uint8_t cc, const std::uint8_t dst) {
        Code.push_back(static_cast<std::uint8_t>(0x40 | ((dst & 8) >> 3)));
        Code.push_back(0x0F);
        Code.push_back(static_cast<std::uint8_t>(0x90 | cc));
        Code.push_back(static_cast<std::uint8_t>(0xC0 | (dst & 7)));
    }

    void Movzx8To32(const std::uint8_t dst, const std::uint8_t src) {
        Code.push_back(static_cast<std::uint8_t>(0x40 | ((dst & 8) >> 1) | ((src & 8) >> 3)));
        Code.push_back(0x0F);
        Code.push_back(0xB6);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((dst & 7) << 3) | (src & 7)));
    }

    void Movsxd(const std::uint8_t dst, const std::uint8_t src) {
        _emitRex(true, dst, 0, src);
        Code.push_back(0x63);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((dst & 7) << 3) | (src & 7)));
    }

    void LoadRsp32(const std::uint8_t dst, const std::int32_t disp) { _memRsp(0x8B, false, dst, disp); }
    void LoadRsp64(const std::uint8_t dst, const std::int32_t disp) { _memRsp(0x8B, true, dst, disp); }
    void StoreRsp32(const std::int32_t disp, const std::uint8_t src) { _memRsp(0x89, false, src, disp); }
    void StoreRsp64(const std::int32_t disp, const std::uint8_t src) { _memRsp(0x89, true, src, disp); }

    void LoadRspIdx32(const std::uint8_t dst, const std::uint8_t idx, const std::int32_t disp) { _memRspIdx(0x8B, false, dst, idx, disp); }
    void StoreRspIdx32(const std::uint8_t idx, const std::int32_t disp, const std::uint8_t src) { _memRspIdx(0x89, false, src, idx, disp); }
    void LoadRspIdx64(const std::uint8_t dst, const std::uint8_t idx, const std::int32_t disp) { _memRspIdx(0x8B, true, dst, idx, disp); }
    void StoreRspIdx64(const std::uint8_t idx, const std::int32_t disp, const std::uint8_t src) { _memRspIdx(0x89, true, src, idx, disp); }

private:
    void _appendU32(const std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8)
            Code.push_back(static_cast<std::uint8_t>(value >> shift));
    }

    void _emitRel32(Label& label) {
        const auto fixup = Code.size();
        if (label.Pos) {
            const auto rel = static_cast<std::int64_t>(*label.Pos) - static_cast<std::int64_t>(fixup + 4);
            if (rel < std::numeric_limits<std::int32_t>::min() || rel > std::numeric_limits<std::int32_t>::max())
                throw CodegenException("FMA branch displacement overflow");
            _appendU32(static_cast<std::uint32_t>(static_cast<std::int32_t>(rel)));
        } else {
            label.Fixups.push_back(fixup);
            _appendU32(0);
        }
    }

    void _emitRex(const bool wide, const std::uint8_t reg, const std::uint8_t index, const std::uint8_t rm) {
        Code.push_back(static_cast<std::uint8_t>(0x40 | (wide ? 8 : 0) | ((reg & 8) >> 1) | ((index & 8) >> 2) | ((rm & 8) >> 3)));
    }

    void _emitRexOpt(const bool wide, const std::uint8_t reg, const std::uint8_t index, const std::uint8_t rm) {
        const auto byte = static_cast<std::uint8_t>(0x40 | (wide ? 8 : 0) | ((reg & 8) >> 1) | ((index & 8) >> 2) | ((rm & 8) >> 3));
        if (byte != 0x40) Code.push_back(byte);
    }

    void _opRr(const std::uint8_t opcode, const bool wide, const std::uint8_t reg, const std::uint8_t rm) {
        _emitRexOpt(wide, reg, 0, rm);
        Code.push_back(opcode);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | ((reg & 7) << 3) | (rm & 7)));
    }

    void _aluImm32(const bool wide, const std::uint8_t ext, const std::uint8_t dst, const std::uint32_t imm) {
        _emitRexOpt(wide, 0, 0, dst);
        Code.push_back(0x81);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | (ext << 3) | (dst & 7)));
        _appendU32(imm);
    }

    void _shiftImm(const bool wide, const std::uint8_t ext, const std::uint8_t dst, const std::uint8_t count) {
        _emitRexOpt(wide, 0, 0, dst);
        Code.push_back(0xC1);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | (ext << 3) | (dst & 7)));
        Code.push_back(count);
    }

    void _shiftCl(const bool wide, const std::uint8_t ext, const std::uint8_t dst) {
        _emitRexOpt(wide, 0, 0, dst);
        Code.push_back(0xD3);
        Code.push_back(static_cast<std::uint8_t>(0xC0 | (ext << 3) | (dst & 7)));
    }

    void _memRsp(const std::uint8_t opcode, const bool wide, const std::uint8_t reg, const std::int32_t disp) {
        _emitRexOpt(wide, reg, 0, Stack);
        Code.push_back(opcode);
        Code.push_back(static_cast<std::uint8_t>(0x84 | ((reg & 7) << 3)));
        Code.push_back(0x24);
        _appendU32(static_cast<std::uint32_t>(disp));
    }

    void _memRspIdx(const std::uint8_t opcode, const bool wide, const std::uint8_t reg, const std::uint8_t idx, const std::int32_t disp) {
        _emitRexOpt(wide, reg, idx, Stack);
        Code.push_back(opcode);
        Code.push_back(static_cast<std::uint8_t>(0x84 | ((reg & 7) << 3)));
        Code.push_back(static_cast<std::uint8_t>(((idx & 7) << 3) | Stack));
        _appendU32(static_cast<std::uint32_t>(disp));
    }
};

const char* _fmaName(const FmaOperation operation, const FmaOrder order, const bool doublePrecision, const bool scalar) {
    const unsigned orderIndex = static_cast<unsigned>(order);
    const unsigned typeIndex = (doublePrecision ? 1u : 0u) + (scalar ? 2u : 0u);
    switch (operation) {
    case FmaOperation::Madd: {
        static constexpr std::array<std::array<const char*, 4>, 3> kNames{{
            {"VFMADD132PS", "VFMADD132PD", "VFMADD132SS", "VFMADD132SD"},
            {"VFMADD213PS", "VFMADD213PD", "VFMADD213SS", "VFMADD213SD"},
            {"VFMADD231PS", "VFMADD231PD", "VFMADD231SS", "VFMADD231SD"},
        }};
        return kNames[orderIndex][typeIndex];
    }
    case FmaOperation::Msub: {
        static constexpr std::array<std::array<const char*, 4>, 3> kNames{{
            {"VFMSUB132PS", "VFMSUB132PD", "VFMSUB132SS", "VFMSUB132SD"},
            {"VFMSUB213PS", "VFMSUB213PD", "VFMSUB213SS", "VFMSUB213SD"},
            {"VFMSUB231PS", "VFMSUB231PD", "VFMSUB231SS", "VFMSUB231SD"},
        }};
        return kNames[orderIndex][typeIndex];
    }
    case FmaOperation::Nmadd: {
        static constexpr std::array<std::array<const char*, 4>, 3> kNames{{
            {"VFNMADD132PS", "VFNMADD132PD", "VFNMADD132SS", "VFNMADD132SD"},
            {"VFNMADD213PS", "VFNMADD213PD", "VFNMADD213SS", "VFNMADD213SD"},
            {"VFNMADD231PS", "VFNMADD231PD", "VFNMADD231SS", "VFNMADD231SD"},
        }};
        return kNames[orderIndex][typeIndex];
    }
    case FmaOperation::Nmsub: {
        static constexpr std::array<std::array<const char*, 4>, 3> kNames{{
            {"VFNMSUB132PS", "VFNMSUB132PD", "VFNMSUB132SS", "VFNMSUB132SD"},
            {"VFNMSUB213PS", "VFNMSUB213PD", "VFNMSUB213SS", "VFNMSUB213SD"},
            {"VFNMSUB231PS", "VFNMSUB231PD", "VFNMSUB231SS", "VFNMSUB231SD"},
        }};
        return kNames[orderIndex][typeIndex];
    }
    case FmaOperation::MaddSub: {
        static constexpr std::array<std::array<const char*, 2>, 3> kNames{{
            {"VFMADDSUB132PS", "VFMADDSUB132PD"},
            {"VFMADDSUB213PS", "VFMADDSUB213PD"},
            {"VFMADDSUB231PS", "VFMADDSUB231PD"},
        }};
        return kNames[orderIndex][doublePrecision ? 1u : 0u];
    }
    case FmaOperation::MsubAdd: {
        static constexpr std::array<std::array<const char*, 2>, 3> kNames{{
            {"VFMSUBADD132PS", "VFMSUBADD132PD"},
            {"VFMSUBADD213PS", "VFMSUBADD213PD"},
            {"VFMSUBADD231PS", "VFMSUBADD231PD"},
        }};
        return kNames[orderIndex][doublePrecision ? 1u : 0u];
    }
    }
    throw CodegenException("unknown FMA operation");
}

std::int32_t _selectFrame(const std::optional<MemoryOperand>& memory, const std::int32_t memBytes) {
    if (!memory || !UnindexedStack(*memory)) return kDefaultFrame;
    const auto displacement = static_cast<std::int64_t>(memory->Displacement);
    if (displacement >= kSpillBytes - kDefaultFrame || displacement + memBytes <= -kDefaultFrame)
        return kDefaultFrame;
    const auto needed = std::max<std::int64_t>(kDefaultFrame, (-displacement + kSpillBytes + 15) & ~std::int64_t{15});
    if (needed > std::numeric_limits<std::int32_t>::max())
        throw CodegenException("stack spill frame exceeds 32-bit displacement");
    return static_cast<std::int32_t>(needed);
}

void _emitUnpack32(Emitter& e, const std::uint8_t absReg, const std::uint8_t expDst, const std::uint8_t mantDst) {
    Label subnormal;
    Label done;
    e.Mov32(expDst, absReg);
    e.ShrImm32(expDst, 23);
    e.Mov32(mantDst, absReg);
    e.AndImm32(mantDst, 0x007FFFFFu);
    e.Test32(expDst, expDst);
    e.Jcc(kCcE, subnormal);
    e.SubImm32(expDst, 127);
    e.OrImm32(mantDst, 0x00800000u);
    e.Jmp(done);
    e.Bind(subnormal);
    e.Bsr32(expDst, mantDst);
    e.MovImm32(kRcx, 23);
    e.Sub32(kRcx, expDst);
    e.ShlCl32(mantDst);
    e.SubImm32(expDst, 149);
    e.Bind(done);
}

void _emitUnpack64(Emitter& e, const std::uint8_t absReg, const std::uint8_t expDst, const std::uint8_t mantDst, const std::uint8_t tmpReg) {
    Label subnormal;
    Label done;
    e.Mov64(expDst, absReg);
    e.ShrImm64(expDst, 52);
    e.MovImm64(tmpReg, 0x000FFFFFFFFFFFFFULL);
    e.Mov64(mantDst, absReg);
    e.And64(mantDst, tmpReg);
    e.Test32(expDst, expDst);
    e.Jcc(kCcE, subnormal);
    e.SubImm32(expDst, 1023);
    e.MovImm64(tmpReg, 0x0010000000000000ULL);
    e.Or64(mantDst, tmpReg);
    e.Jmp(done);
    e.Bind(subnormal);
    e.Bsr64(expDst, mantDst);
    e.MovImm32(kRcx, 52);
    e.Sub32(kRcx, expDst);
    e.ShlCl64(mantDst);
    e.SubImm32(expDst, 1074);
    e.Bind(done);
}

void _emitFma32Core(Emitter& e) {
    Label done;
    Label noDaz;
    e.TestImm32(kR14, 0x0040u);
    e.Jcc(kCcE, noDaz);
    for (const auto reg : {kRdi, kRsi, kRdx}) {
        Label keep;
        e.TestImm32(reg, 0x7F800000u);
        e.Jcc(kCcNe, keep);
        e.AndImm32(reg, 0x80000000u);
        e.Bind(keep);
    }
    e.Bind(noDaz);

    for (const auto reg : {kRdi, kRsi, kRdx}) {
        Label notNan;
        e.Mov32(kRax, reg);
        e.AndImm32(kRax, 0x7FFFFFFFu);
        e.CmpImm32(kRax, static_cast<std::int32_t>(0x7F800000u));
        e.Jcc(kCcBe, notNan);
        e.Mov32(kRax, reg);
        e.OrImm32(kRax, 0x00400000u);
        e.Jmp(done);
        e.Bind(notNan);
    }

    e.Mov32(kR8, kRdi);
    e.Xor32(kR8, kRsi);
    e.ShrImm32(kR8, 31);
    e.Mov32(kRax, kR13);
    e.AndImm32(kRax, 1u);
    e.Xor32(kR8, kRax);

    e.Mov32(kR9, kRdx);
    e.ShrImm32(kR9, 31);
    e.Mov32(kRax, kR13);
    e.ShrImm32(kRax, 1);
    e.AndImm32(kRax, 1u);
    e.Xor32(kR9, kRax);

    e.AndImm32(kRdi, 0x7FFFFFFFu);
    e.AndImm32(kRsi, 0x7FFFFFFFu);
    e.AndImm32(kRdx, 0x7FFFFFFFu);

    Label returnIndef;
    Label prodInf;
    Label checkBInf;
    Label checkCInf;
    Label allFinite;

    e.CmpImm32(kRdi, static_cast<std::int32_t>(0x7F800000u));
    e.Jcc(kCcNe, checkBInf);
    e.Test32(kRsi, kRsi);
    e.Jcc(kCcE, returnIndef);
    e.Jmp(prodInf);

    e.Bind(checkBInf);
    e.CmpImm32(kRsi, static_cast<std::int32_t>(0x7F800000u));
    e.Jcc(kCcNe, checkCInf);
    e.Test32(kRdi, kRdi);
    e.Jcc(kCcE, returnIndef);

    e.Bind(prodInf);
    e.CmpImm32(kRdx, static_cast<std::int32_t>(0x7F800000u));
    Label retProdInf;
    e.Jcc(kCcNe, retProdInf);
    e.Cmp32(kR8, kR9);
    e.Jcc(kCcNe, returnIndef);
    e.Bind(retProdInf);
    e.Mov32(kRax, kR8);
    e.ShlImm32(kRax, 31);
    e.OrImm32(kRax, 0x7F800000u);
    e.Jmp(done);

    e.Bind(checkCInf);
    e.CmpImm32(kRdx, static_cast<std::int32_t>(0x7F800000u));
    e.Jcc(kCcNe, allFinite);
    e.Mov32(kRax, kR9);
    e.ShlImm32(kRax, 31);
    e.OrImm32(kRax, 0x7F800000u);
    e.Jmp(done);

    e.Bind(returnIndef);
    e.MovImm32(kRax, 0xFFC00000u);
    e.Jmp(done);

    e.Bind(allFinite);
    Label prodNonZero;
    Label normalize;
    Label cancelZero;

    e.Test32(kRdi, kRdi);
    Label prodIsZero;
    e.Jcc(kCcE, prodIsZero);
    e.Test32(kRsi, kRsi);
    e.Jcc(kCcNe, prodNonZero);

    e.Bind(prodIsZero);
    e.Test32(kRdx, kRdx);
    Label onlyC;
    e.Jcc(kCcNe, onlyC);
    e.Cmp32(kR8, kR9);
    e.Jcc(kCcNe, cancelZero);
    e.Mov32(kRax, kR8);
    e.ShlImm32(kRax, 31);
    e.Jmp(done);

    e.Bind(onlyC);
    _emitUnpack32(e, kRdx, kRbp, kR10);
    e.ShlImm64(kR10, 38);
    e.Xor32(kR11, kR11);
    e.Mov32(kR8, kR9);
    e.Jmp(normalize);

    e.Bind(prodNonZero);
    _emitUnpack32(e, kRdi, kRax, kRbx);
    e.Mov32(kRbp, kRax);
    _emitUnpack32(e, kRsi, kRax, kR10);
    e.Add32(kRbp, kRax);
    e.Imul64(kR10, kRbx);
    e.MovImm64(kRax, 1ULL << 47);
    e.Test64(kR10, kRax);
    Label mpNorm;
    Label mpReady;
    e.Jcc(kCcNe, mpNorm);
    e.ShlImm64(kR10, 1);
    e.Jmp(mpReady);
    e.Bind(mpNorm);
    e.AddImm32(kRbp, 1);
    e.Bind(mpReady);

    e.Test32(kRdx, kRdx);
    Label bothNonZero;
    e.Jcc(kCcNe, bothNonZero);
    e.ShlImm64(kR10, 14);
    e.Xor32(kR11, kR11);
    e.Jmp(normalize);

    e.Bind(bothNonZero);
    _emitUnpack32(e, kRdx, kR12, kR11);
    e.ShlImm64(kR11, 24);

    Label noSwap;
    Label doSwap;
    e.Cmp32(kR12, kRbp);
    e.Jcc(kCcG, doSwap);
    e.Jcc(kCcL, noSwap);
    e.Cmp64(kR11, kR10);
    e.Jcc(kCcBe, noSwap);

    e.Bind(doSwap);
    e.Mov64(kRax, kR10);
    e.Mov64(kR10, kR11);
    e.Mov64(kR11, kRax);
    e.Mov32(kRax, kRbp);
    e.Mov32(kRbp, kR12);
    e.Mov32(kR12, kRax);
    e.Mov32(kRax, kR8);
    e.Mov32(kR8, kR9);
    e.Mov32(kR9, kRax);

    e.Bind(noSwap);
    e.Mov32(kRcx, kRbp);
    e.Sub32(kRcx, kR12);
    e.ShlImm64(kR10, 14);
    e.Xor32(kRax, kRax);
    e.ShlImm64(kR11, 14);
    e.Xor32(kRdx, kRdx);

    Label diffHuge;
    Label diffDone;
    Label diffLt64;
    e.CmpImm32(kRcx, 127);
    e.Jcc(kCcAe, diffHuge);
    e.Xor32(kRbx, kRbx);
    e.CmpImm32(kRcx, 64);
    e.Jcc(kCcB, diffLt64);
    e.Or64(kRbx, kRdx);
    e.Mov64(kRdx, kR11);
    e.Xor32(kR11, kR11);
    e.SubImm32(kRcx, 64);
    e.Bind(diffLt64);
    e.Xor32(kRdi, kRdi);
    e.ShrdCl64(kRdi, kRdx);
    e.Or64(kRbx, kRdi);
    e.ShrdCl64(kRdx, kR11);
    e.ShrCl64(kR11);
    e.Test64(kRbx, kRbx);
    e.Jcc(kCcE, diffDone);
    e.OrImm64(kRdx, 1);
    e.Jmp(diffDone);

    e.Bind(diffHuge);
    e.Xor32(kR11, kR11);
    e.MovImm32(kRdx, 1);

    e.Bind(diffDone);
    Label doSub;
    e.Cmp32(kR8, kR9);
    e.Jcc(kCcNe, doSub);
    e.Add64(kRax, kRdx);
    e.Adc64(kR10, kR11);
    e.Mov64(kR11, kRax);
    e.Jmp(normalize);

    e.Bind(doSub);
    e.Sub64(kRax, kRdx);
    e.Sbb64(kR10, kR11);
    e.Mov64(kR11, kRax);
    e.Mov64(kRdx, kR10);
    e.Or64(kRdx, kR11);
    e.Jcc(kCcNe, normalize);

    e.Bind(cancelZero);
    e.Mov32(kRcx, kR14);
    e.ShrImm32(kRcx, 13);
    e.AndImm32(kRcx, 3u);
    e.Xor32(kRax, kRax);
    e.CmpImm32(kRcx, 1);
    e.Jcc(kCcNe, done);
    e.MovImm32(kRax, 0x80000000u);
    e.Jmp(done);

    e.Bind(normalize);
    Label w1NonZero;
    e.Test64(kR10, kR10);
    e.Jcc(kCcNe, w1NonZero);
    e.Mov64(kR10, kR11);
    e.Xor32(kR11, kR11);
    e.SubImm32(kRbp, 64);
    e.Bind(w1NonZero);

    e.Bsr64(kRax, kR10);
    e.Add32(kRbp, kRax);
    e.SubImm32(kRbp, 61);
    e.MovImm32(kRcx, 63);
    e.Sub32(kRcx, kRax);
    e.ShldCl64(kR10, kR11);
    e.ShlCl64(kR11);
    Label w0Zero;
    e.Test64(kR11, kR11);
    e.Jcc(kCcE, w0Zero);
    e.OrImm64(kR10, 1);
    e.Bind(w0Zero);

    Label roundStep;
    e.Xor32(kRdi, kRdi);
    e.CmpImm32(kRbp, -126);
    e.Jcc(kCcGe, roundStep);
    e.TestImm32(kR14, 0x8000u);
    e.Jcc(kCcNe, roundStep);
    e.MovImm32(kRdi, 1);
    e.MovImm32(kRcx, static_cast<std::uint32_t>(-126));
    e.Sub32(kRcx, kRbp);
    Label subShiftSmall;
    e.CmpImm32(kRcx, 26);
    e.Jcc(kCcB, subShiftSmall);
    e.MovImm32(kR10, 1);
    e.Jmp(roundStep);
    e.Bind(subShiftSmall);
    e.Xor32(kRax, kRax);
    e.ShrdCl64(kRax, kR10);
    e.ShrCl64(kR10);
    e.Test64(kRax, kRax);
    e.Jcc(kCcE, roundStep);
    e.OrImm64(kR10, 1);

    e.Bind(roundStep);
    e.Mov64(kRax, kR10);
    e.ShrImm64(kRax, 40);
    e.Mov64(kRdx, kR10);
    e.ShrImm64(kRdx, 39);
    e.AndImm32(kRdx, 1u);
    e.MovImm64(kRbx, 0x0000007FFFFFFFFFULL);
    e.Test64(kR10, kRbx);
    e.Setcc(kCcNe, kRbx);
    e.Movzx8To32(kRbx, kRbx);

    e.Mov32(kRcx, kR14);
    e.ShrImm32(kRcx, 13);
    e.AndImm32(kRcx, 3u);

    Label roundDone;
    Label rcNotRn;
    Label rcNotRd;
    Label rcNotRu;

    e.Test32(kRcx, kRcx);
    e.Jcc(kCcNe, rcNotRn);
    e.Test32(kRdx, kRdx);
    e.Jcc(kCcE, roundDone);
    e.Test32(kRbx, kRbx);
    Label doInc;
    e.Jcc(kCcNe, doInc);
    e.TestImm32(kRax, 1u);
    e.Jcc(kCcE, roundDone);
    e.Jmp(doInc);

    e.Bind(rcNotRn);
    e.CmpImm32(kRcx, 1);
    e.Jcc(kCcNe, rcNotRd);
    e.Test32(kR8, kR8);
    e.Jcc(kCcE, roundDone);
    e.Or32(kRdx, kRbx);
    e.Jcc(kCcNe, doInc);
    e.Jmp(roundDone);

    e.Bind(rcNotRd);
    e.CmpImm32(kRcx, 2);
    e.Jcc(kCcNe, rcNotRu);
    e.Test32(kR8, kR8);
    e.Jcc(kCcNe, roundDone);
    e.Or32(kRdx, kRbx);
    e.Jcc(kCcNe, doInc);
    e.Jmp(roundDone);

    e.Bind(rcNotRu);
    e.Jmp(roundDone);

    e.Bind(doInc);
    e.AddImm64(kRax, 1);

    e.Bind(roundDone);
    Label packNormal;
    e.Test32(kRdi, kRdi);
    e.Jcc(kCcE, packNormal);
    e.ShlImm32(kR8, 31);
    e.Or32(kRax, kR8);
    e.Jmp(done);

    e.Bind(packNormal);
    Label noCarry;
    e.CmpImm32(kRax, 1 << 24);
    e.Jcc(kCcNe, noCarry);
    e.ShrImm32(kRax, 1);
    e.AddImm32(kRbp, 1);
    e.Bind(noCarry);

    Label notFtzUnderflow;
    e.CmpImm32(kRbp, -126);
    e.Jcc(kCcGe, notFtzUnderflow);
    e.Mov32(kRax, kR8);
    e.ShlImm32(kRax, 31);
    e.Jmp(done);

    e.Bind(notFtzUnderflow);
    Label notOverflow;
    e.CmpImm32(kRbp, 127);
    e.Jcc(kCcLe, notOverflow);
    Label ovfInf;
    Label ovfMax;
    e.Test32(kRcx, kRcx);
    e.Jcc(kCcE, ovfInf);
    e.CmpImm32(kRcx, 1);
    Label ovfCheckRu;
    e.Jcc(kCcNe, ovfCheckRu);
    e.Test32(kR8, kR8);
    e.Jcc(kCcNe, ovfInf);
    e.Jmp(ovfMax);
    e.Bind(ovfCheckRu);
    e.CmpImm32(kRcx, 2);
    e.Jcc(kCcNe, ovfMax);
    e.Test32(kR8, kR8);
    e.Jcc(kCcE, ovfInf);
    e.Bind(ovfMax);
    e.Mov32(kRax, kR8);
    e.ShlImm32(kRax, 31);
    e.OrImm32(kRax, 0x7F7FFFFFu);
    e.Jmp(done);
    e.Bind(ovfInf);
    e.Mov32(kRax, kR8);
    e.ShlImm32(kRax, 31);
    e.OrImm32(kRax, 0x7F800000u);
    e.Jmp(done);

    e.Bind(notOverflow);
    e.AddImm32(kRbp, 127);
    e.ShlImm32(kRbp, 23);
    e.AndImm32(kRax, 0x007FFFFFu);
    e.Or32(kRax, kRbp);
    e.ShlImm32(kR8, 31);
    e.Or32(kRax, kR8);

    e.Bind(done);
}

void _emitFma64Core(Emitter& e) {
    Label done;
    Label noDaz;
    e.TestImm32(kR14, 0x0040u);
    e.Jcc(kCcE, noDaz);
    e.MovImm64(kR10, 0x7FF0000000000000ULL);
    e.MovImm64(kR11, 0x8000000000000000ULL);
    for (const auto reg : {kRdi, kRsi, kRdx}) {
        Label keep;
        e.Test64(reg, kR10);
        e.Jcc(kCcNe, keep);
        e.And64(reg, kR11);
        e.Bind(keep);
    }
    e.Bind(noDaz);

    e.MovImm64(kR10, 0x7FFFFFFFFFFFFFFFULL);
    e.MovImm64(kR11, 0x7FF0000000000000ULL);
    e.MovImm64(kR12, 0x0008000000000000ULL);
    for (const auto reg : {kRdi, kRsi, kRdx}) {
        Label notNan;
        e.Mov64(kRax, reg);
        e.And64(kRax, kR10);
        e.Cmp64(kRax, kR11);
        e.Jcc(kCcBe, notNan);
        e.Mov64(kRax, reg);
        e.Or64(kRax, kR12);
        e.Jmp(done);
        e.Bind(notNan);
    }

    e.Mov64(kR8, kRdi);
    e.Xor64(kR8, kRsi);
    e.ShrImm64(kR8, 63);
    e.Mov32(kRax, kR13);
    e.AndImm32(kRax, 1u);
    e.Xor32(kR8, kRax);

    e.Mov64(kR9, kRdx);
    e.ShrImm64(kR9, 63);
    e.Mov32(kRax, kR13);
    e.ShrImm32(kRax, 1);
    e.AndImm32(kRax, 1u);
    e.Xor32(kR9, kRax);

    e.And64(kRdi, kR10);
    e.And64(kRsi, kR10);
    e.And64(kRdx, kR10);

    Label returnIndef;
    Label prodInf;
    Label checkBInf;
    Label checkCInf;
    Label allFinite;

    e.Cmp64(kRdi, kR11);
    e.Jcc(kCcNe, checkBInf);
    e.Test64(kRsi, kRsi);
    e.Jcc(kCcE, returnIndef);
    e.Jmp(prodInf);

    e.Bind(checkBInf);
    e.Cmp64(kRsi, kR11);
    e.Jcc(kCcNe, checkCInf);
    e.Test64(kRdi, kRdi);
    e.Jcc(kCcE, returnIndef);

    e.Bind(prodInf);
    e.Cmp64(kRdx, kR11);
    Label retProdInf;
    e.Jcc(kCcNe, retProdInf);
    e.Cmp32(kR8, kR9);
    e.Jcc(kCcNe, returnIndef);
    e.Bind(retProdInf);
    e.Mov64(kRax, kR8);
    e.ShlImm64(kRax, 63);
    e.Or64(kRax, kR11);
    e.Jmp(done);

    e.Bind(checkCInf);
    e.Cmp64(kRdx, kR11);
    e.Jcc(kCcNe, allFinite);
    e.Mov64(kRax, kR9);
    e.ShlImm64(kRax, 63);
    e.Or64(kRax, kR11);
    e.Jmp(done);

    e.Bind(returnIndef);
    e.MovImm64(kRax, 0xFFF8000000000000ULL);
    e.Jmp(done);

    e.Bind(allFinite);
    Label prodNonZero;
    Label normalize;
    Label cancelZero;

    e.Test64(kRdi, kRdi);
    Label prodIsZero;
    e.Jcc(kCcE, prodIsZero);
    e.Test64(kRsi, kRsi);
    e.Jcc(kCcNe, prodNonZero);

    e.Bind(prodIsZero);
    e.Test64(kRdx, kRdx);
    Label onlyC;
    e.Jcc(kCcNe, onlyC);
    e.Cmp32(kR8, kR9);
    e.Jcc(kCcNe, cancelZero);
    e.Mov64(kRax, kR8);
    e.ShlImm64(kRax, 63);
    e.Jmp(done);

    e.Bind(onlyC);
    _emitUnpack64(e, kRdx, kRbp, kR10, kRax);
    e.ShlImm64(kR10, 9);
    e.Xor32(kR11, kR11);
    e.Xor32(kR12, kR12);
    e.Mov32(kR8, kR9);
    e.Jmp(normalize);

    e.Bind(prodNonZero);
    e.StoreRsp64(kLocal0, kRdx);
    _emitUnpack64(e, kRdi, kRax, kRbx, kR10);
    e.StoreRsp32(kLocal1, kRax);
    _emitUnpack64(e, kRsi, kRbp, kRax, kR10);
    e.LoadRsp32(kR10, kLocal1);
    e.Add32(kRbp, kR10);
    e.Mul64(kRbx);

    e.MovImm64(kRcx, 1ULL << 41);
    e.Test64(kRdx, kRcx);
    Label pNorm;
    Label pReady;
    e.Jcc(kCcNe, pNorm);
    e.ShldImm64(kRdx, kRax, 1);
    e.ShlImm64(kRax, 1);
    e.Jmp(pReady);
    e.Bind(pNorm);
    e.AddImm32(kRbp, 1);
    e.Bind(pReady);

    e.ShldImm64(kRdx, kRax, 20);
    e.Mov64(kR10, kRdx);
    e.ShlImm64(kRax, 20);
    e.Mov64(kR11, kRax);
    e.Xor32(kR12, kR12);

    e.LoadRsp64(kRdx, kLocal0);
    e.Test64(kRdx, kRdx);
    e.Jcc(kCcE, normalize);

    _emitUnpack64(e, kRdx, kRsi, kRdi, kRax);
    e.ShlImm64(kRdi, 9);
    e.Xor32(kRbx, kRbx);

    Label noSwap;
    Label doSwap;
    e.Cmp32(kRsi, kRbp);
    e.Jcc(kCcG, doSwap);
    e.Jcc(kCcL, noSwap);
    e.Cmp64(kRdi, kR10);
    e.Jcc(kCcBe, noSwap);

    e.Bind(doSwap);
    e.Mov64(kRax, kR10);
    e.Mov64(kR10, kRdi);
    e.Mov64(kRdi, kRax);
    e.Mov64(kRbx, kR11);
    e.Xor32(kR11, kR11);
    e.Mov32(kRax, kRbp);
    e.Mov32(kRbp, kRsi);
    e.Mov32(kRsi, kRax);
    e.Mov32(kRax, kR8);
    e.Mov32(kR8, kR9);
    e.Mov32(kR9, kRax);

    e.Bind(noSwap);
    e.Mov32(kRcx, kRbp);
    e.Sub32(kRcx, kRsi);
    e.Xor32(kRdx, kRdx);

    Label diffHuge;
    Label diffDone;
    Label diffLt64;
    e.CmpImm32(kRcx, 191);
    e.Jcc(kCcAe, diffHuge);
    e.Xor32(kRsi, kRsi);
    e.CmpImm32(kRcx, 64);
    e.Jcc(kCcB, diffLt64);
    e.Or64(kRsi, kRdx);
    e.Mov64(kRdx, kRbx);
    e.Mov64(kRbx, kRdi);
    e.Xor32(kRdi, kRdi);
    e.SubImm32(kRcx, 64);
    e.CmpImm32(kRcx, 64);
    e.Jcc(kCcB, diffLt64);
    e.Or64(kRsi, kRdx);
    e.Mov64(kRdx, kRbx);
    e.Xor32(kRbx, kRbx);
    e.SubImm32(kRcx, 64);
    e.Bind(diffLt64);
    e.Xor32(kRax, kRax);
    e.ShrdCl64(kRax, kRdx);
    e.Or64(kRsi, kRax);
    e.ShrdCl64(kRdx, kRbx);
    e.ShrdCl64(kRbx, kRdi);
    e.ShrCl64(kRdi);
    e.Test64(kRsi, kRsi);
    e.Jcc(kCcE, diffDone);
    e.OrImm64(kRdx, 1);
    e.Jmp(diffDone);

    e.Bind(diffHuge);
    e.Xor32(kRdi, kRdi);
    e.Xor32(kRbx, kRbx);
    e.MovImm32(kRdx, 1);

    e.Bind(diffDone);
    Label doSub;
    e.Cmp32(kR8, kR9);
    e.Jcc(kCcNe, doSub);
    e.Add64(kR12, kRdx);
    e.Adc64(kR11, kRbx);
    e.Adc64(kR10, kRdi);
    e.Jmp(normalize);

    e.Bind(doSub);
    e.Sub64(kR12, kRdx);
    e.Sbb64(kR11, kRbx);
    e.Sbb64(kR10, kRdi);
    e.Mov64(kRax, kR10);
    e.Or64(kRax, kR11);
    e.Or64(kRax, kR12);
    e.Jcc(kCcNe, normalize);

    e.Bind(cancelZero);
    e.Mov32(kRcx, kR14);
    e.ShrImm32(kRcx, 13);
    e.AndImm32(kRcx, 3u);
    e.Xor32(kRax, kRax);
    e.CmpImm32(kRcx, 1);
    e.Jcc(kCcNe, done);
    e.MovImm64(kRax, 0x8000000000000000ULL);
    e.Jmp(done);

    e.Bind(normalize);
    Label w2NonZero;
    e.Test64(kR10, kR10);
    e.Jcc(kCcNe, w2NonZero);
    Label w1Zero;
    e.Test64(kR11, kR11);
    e.Jcc(kCcE, w1Zero);
    e.Mov64(kR10, kR11);
    e.Mov64(kR11, kR12);
    e.Xor32(kR12, kR12);
    e.SubImm32(kRbp, 64);
    e.Jmp(w2NonZero);
    e.Bind(w1Zero);
    e.Mov64(kR10, kR12);
    e.Xor32(kR11, kR11);
    e.Xor32(kR12, kR12);
    e.SubImm32(kRbp, 128);
    e.Bind(w2NonZero);

    e.Bsr64(kRax, kR10);
    e.Add32(kRbp, kRax);
    e.SubImm32(kRbp, 61);
    e.MovImm32(kRcx, 63);
    e.Sub32(kRcx, kRax);
    e.ShldCl64(kR10, kR11);
    e.ShldCl64(kR11, kR12);
    e.ShlCl64(kR12);
    e.Or64(kR11, kR12);
    Label lowZero;
    e.Jcc(kCcE, lowZero);
    e.OrImm64(kR10, 1);
    e.Bind(lowZero);

    Label roundStep;
    e.Xor32(kRdi, kRdi);
    e.CmpImm32(kRbp, -1022);
    e.Jcc(kCcGe, roundStep);
    e.TestImm32(kR14, 0x8000u);
    e.Jcc(kCcNe, roundStep);
    e.MovImm32(kRdi, 1);
    e.MovImm32(kRcx, static_cast<std::uint32_t>(-1022));
    e.Sub32(kRcx, kRbp);
    Label subShiftSmall;
    e.CmpImm32(kRcx, 55);
    e.Jcc(kCcB, subShiftSmall);
    e.MovImm32(kR10, 1);
    e.Jmp(roundStep);
    e.Bind(subShiftSmall);
    e.Xor32(kRax, kRax);
    e.ShrdCl64(kRax, kR10);
    e.ShrCl64(kR10);
    e.Test64(kRax, kRax);
    e.Jcc(kCcE, roundStep);
    e.OrImm64(kR10, 1);

    e.Bind(roundStep);
    e.Mov64(kRax, kR10);
    e.ShrImm64(kRax, 11);
    e.Mov64(kRdx, kR10);
    e.ShrImm64(kRdx, 10);
    e.AndImm32(kRdx, 1u);
    e.TestImm64(kR10, 0x3FF);
    e.Setcc(kCcNe, kRbx);
    e.Movzx8To32(kRbx, kRbx);

    e.Mov32(kRcx, kR14);
    e.ShrImm32(kRcx, 13);
    e.AndImm32(kRcx, 3u);

    Label roundDone;
    Label rcNotRn;
    Label rcNotRd;
    Label rcNotRu;
    Label doInc;

    e.Test32(kRcx, kRcx);
    e.Jcc(kCcNe, rcNotRn);
    e.Test32(kRdx, kRdx);
    e.Jcc(kCcE, roundDone);
    e.Test32(kRbx, kRbx);
    e.Jcc(kCcNe, doInc);
    e.TestImm32(kRax, 1u);
    e.Jcc(kCcE, roundDone);
    e.Jmp(doInc);

    e.Bind(rcNotRn);
    e.CmpImm32(kRcx, 1);
    e.Jcc(kCcNe, rcNotRd);
    e.Test32(kR8, kR8);
    e.Jcc(kCcE, roundDone);
    e.Or32(kRdx, kRbx);
    e.Jcc(kCcNe, doInc);
    e.Jmp(roundDone);

    e.Bind(rcNotRd);
    e.CmpImm32(kRcx, 2);
    e.Jcc(kCcNe, rcNotRu);
    e.Test32(kR8, kR8);
    e.Jcc(kCcNe, roundDone);
    e.Or32(kRdx, kRbx);
    e.Jcc(kCcNe, doInc);
    e.Jmp(roundDone);

    e.Bind(rcNotRu);
    e.Jmp(roundDone);

    e.Bind(doInc);
    e.AddImm64(kRax, 1);

    e.Bind(roundDone);
    Label packNormal;
    e.Test32(kRdi, kRdi);
    e.Jcc(kCcE, packNormal);
    e.ShlImm64(kR8, 63);
    e.Or64(kRax, kR8);
    e.Jmp(done);

    e.Bind(packNormal);
    Label noCarry;
    e.MovImm64(kRdx, 1ULL << 53);
    e.Cmp64(kRax, kRdx);
    e.Jcc(kCcNe, noCarry);
    e.ShrImm64(kRax, 1);
    e.AddImm32(kRbp, 1);
    e.Bind(noCarry);

    Label notFtzUnderflow;
    e.CmpImm32(kRbp, -1022);
    e.Jcc(kCcGe, notFtzUnderflow);
    e.Mov64(kRax, kR8);
    e.ShlImm64(kRax, 63);
    e.Jmp(done);

    e.Bind(notFtzUnderflow);
    Label notOverflow;
    e.CmpImm32(kRbp, 1023);
    e.Jcc(kCcLe, notOverflow);
    Label ovfInf;
    Label ovfMax;
    e.Test32(kRcx, kRcx);
    e.Jcc(kCcE, ovfInf);
    e.CmpImm32(kRcx, 1);
    Label ovfCheckRu;
    e.Jcc(kCcNe, ovfCheckRu);
    e.Test32(kR8, kR8);
    e.Jcc(kCcNe, ovfInf);
    e.Jmp(ovfMax);
    e.Bind(ovfCheckRu);
    e.CmpImm32(kRcx, 2);
    e.Jcc(kCcNe, ovfMax);
    e.Test32(kR8, kR8);
    e.Jcc(kCcE, ovfInf);
    e.Bind(ovfMax);
    e.MovImm64(kRax, 0x7FEFFFFFFFFFFFFFULL);
    e.ShlImm64(kR8, 63);
    e.Or64(kRax, kR8);
    e.Jmp(done);
    e.Bind(ovfInf);
    e.MovImm64(kRax, 0x7FF0000000000000ULL);
    e.ShlImm64(kR8, 63);
    e.Or64(kRax, kR8);
    e.Jmp(done);

    e.Bind(notOverflow);
    e.AddImm32(kRbp, 1023);
    e.Movsxd(kRdx, kRbp);
    e.ShlImm64(kRdx, 52);
    e.MovImm64(kRcx, 0x000FFFFFFFFFFFFFULL);
    e.And64(kRax, kRcx);
    e.Or64(kRax, kRdx);
    e.ShlImm64(kR8, 63);
    e.Or64(kRax, kR8);

    e.Bind(done);
}

std::uint32_t _initialOpFlags(const FmaOperation operation) {
    switch (operation) {
    case FmaOperation::Madd: return 0u;
    case FmaOperation::Msub: return 2u;
    case FmaOperation::Nmadd: return 1u;
    case FmaOperation::Nmsub: return 3u;
    case FmaOperation::MaddSub: return 2u;
    case FmaOperation::MsubAdd: return 0u;
    }
    throw CodegenException("unknown FMA operation");
}

}

std::optional<FmaOperands> DecodeFma(const std::uint8_t* data, const std::size_t length) {
    const auto instruction = Decode(data, length);
    if (!instruction || instruction->Map != Map0F38 || instruction->Prefix != Prefix66)
        return std::nullopt;
    const auto opcode = instruction->Opcode;
    const auto high = static_cast<std::uint8_t>(opcode >> 4);
    const auto low = static_cast<std::uint8_t>(opcode & 0x0F);
    if (high < 9 || high > 11 || low < 6)
        return std::nullopt;
    const auto order = high == 9 ? FmaOrder::Order132 : (high == 10 ? FmaOrder::Order213 : FmaOrder::Order231);
    FmaOperation operation = FmaOperation::Madd;
    bool scalar = false;
    switch (low) {
    case 0x6: operation = FmaOperation::MaddSub; break;
    case 0x7: operation = FmaOperation::MsubAdd; break;
    case 0x8: operation = FmaOperation::Madd; break;
    case 0x9: operation = FmaOperation::Madd; scalar = true; break;
    case 0xA: operation = FmaOperation::Msub; break;
    case 0xB: operation = FmaOperation::Msub; scalar = true; break;
    case 0xC: operation = FmaOperation::Nmadd; break;
    case 0xD: operation = FmaOperation::Nmadd; scalar = true; break;
    case 0xE: operation = FmaOperation::Nmsub; break;
    case 0xF: operation = FmaOperation::Nmsub; scalar = true; break;
    default: return std::nullopt;
    }
    const auto* name = _fmaName(operation, order, instruction->Wide, scalar);
    if (instruction->End != length)
        throw CodegenException(std::string(name) + " has trailing bytes");
    return FmaOperands{
        name,
        operation,
        order,
        instruction->Wide,
        scalar,
        !scalar && instruction->Vector256,
        instruction->Reg,
        instruction->Vvvv,
        instruction->Rm,
        instruction->Memory,
    };
}

void EmitFma(StubBodyBuilder& body, const FmaOperands& operands) {
    const std::int32_t elemBytes = operands.DoublePrecision ? 8 : 4;
    const std::int32_t lanes = operands.Scalar ? 1 : (operands.Vector256 ? 32 : 16) / elemBytes;
    const std::int32_t memBytes = operands.Scalar ? elemBytes : (operands.Vector256 ? 32 : 16);
    const auto frame = _selectFrame(operands.Memory, memBytes);

    StackSlot(body, 0x8D, true, Stack, 8 - frame);
    body.Raw(std::array{kPushFlags});

    const Form vecStore = operands.Vector256 ? MoveUnaligned256Store : kMoveUnaligned128Store;
    const Form vecLoad = operands.Vector256 ? MoveUnaligned256Load : MoveUnaligned128Load;

    if (operands.Memory) {
        std::uint8_t scratch = 0;
        while (scratch == operands.Op1 || scratch == operands.Op2)
            ++scratch;
        StackOperand(body, MoveUnaligned256Store, scratch, 0, kSaveYmm);
        const Form memLoad = operands.Scalar
            ? (operands.DoublePrecision ? kLoadScalarDouble : kLoadScalarSingle)
            : vecLoad;
        Memory(body, memLoad, scratch, 0, *operands.Memory, frame, operands.Name);
        StackOperand(body, vecStore, scratch, 0, kBufOp3);
        StackOperand(body, MoveUnaligned256Load, scratch, 0, kSaveYmm);
    } else {
        StackOperand(body, vecStore, operands.Op3, 0, kBufOp3);
    }

    StackOperand(body, vecStore, operands.Op1, 0, kBufOp1);
    StackOperand(body, vecStore, operands.Op2, 0, kBufOp2);
    StackOperand(body, kStoreMxcsr, 3, 0, kSlotMxcsr);

    Emitter e;
    for (std::size_t i = 0; i < kSavedGprs.size(); ++i)
        e.StoreRsp64(kSlotGprBase + static_cast<std::int32_t>(i * 8), kSavedGprs[i]);

    e.LoadRsp32(kR14, kSlotMxcsr);
    e.MovImm32(kR13, _initialOpFlags(operands.Operation));
    e.Xor32(kR15, kR15);

    std::int32_t slotA = kBufOp1;
    std::int32_t slotB = kBufOp3;
    std::int32_t slotC = kBufOp2;
    if (operands.Order == FmaOrder::Order213) {
        slotA = kBufOp2;
        slotB = kBufOp1;
        slotC = kBufOp3;
    } else if (operands.Order == FmaOrder::Order231) {
        slotA = kBufOp2;
        slotB = kBufOp3;
        slotC = kBufOp1;
    }

    Label loopStart;
    e.Bind(loopStart);

    if (operands.DoublePrecision) {
        e.LoadRspIdx64(kRdi, kR15, slotA);
        e.LoadRspIdx64(kRsi, kR15, slotB);
        e.LoadRspIdx64(kRdx, kR15, slotC);
        _emitFma64Core(e);
        e.StoreRspIdx64(kR15, kBufOp1, kRax);
    } else {
        e.LoadRspIdx32(kRdi, kR15, slotA);
        e.LoadRspIdx32(kRsi, kR15, slotB);
        e.LoadRspIdx32(kRdx, kR15, slotC);
        _emitFma32Core(e);
        e.StoreRspIdx32(kR15, kBufOp1, kRax);
    }

    if (lanes > 1) {
        if (operands.Operation == FmaOperation::MaddSub || operands.Operation == FmaOperation::MsubAdd)
            e.XorImm32(kR13, 2u);
        e.AddImm64(kR15, elemBytes);
        e.CmpImm64(kR15, lanes * elemBytes);
        e.Jcc(kCcB, loopStart);
    }

    for (std::size_t i = 0; i < kSavedGprs.size(); ++i)
        e.LoadRsp64(kSavedGprs[i], kSlotGprBase + static_cast<std::int32_t>(i * 8));

    body.Raw(e.Code);

    StackOperand(body, vecLoad, operands.Op1, 0, kBufOp1);
    body.Raw(std::array{kPopFlags});
    StackSlot(body, 0x8D, true, Stack, frame - 8);
}

}
