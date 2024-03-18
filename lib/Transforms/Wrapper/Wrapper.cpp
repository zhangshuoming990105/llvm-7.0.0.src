//===- Wrapper.cpp - Transform call to original function to wrapper function ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// 
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <set>

using namespace llvm;

#define DEBUG_TYPE "wrapper"

namespace {
  struct WrapperPass : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    std::set<StringRef> funcs;

    WrapperPass() : ModulePass(ID) {}
    bool runOnModule(Module &M) override;
    void initialize(Module &M);
    bool addWrapper(Function &F);
    bool setDevice(Function &F);
  };
}

bool WrapperPass::runOnModule(Module &M) {
    bool ret = false;
    initialize(M);
    for (auto it = M.begin(), ie = M.end(); it != ie; ++it) {
        Function &func = *it;
        ret |= addWrapper(func);
    }
    return ret;
}

void WrapperPass::initialize(Module &M) {
    for (auto it = M.begin(), ie = M.end(); it != ie; ++it) {
        Function &func = *it;
        StringRef fname = func.getName();
        if (fname.endswith("_wrapper")) {
            FunctionType *type = cast<FunctionType>(func.getType()->getPointerElementType());
            if (type->getNumParams() < 2) continue;
            Type *Arg0 = type->getParamType(0);
            if (Arg0->isPointerTy() && cast<PointerType>(Arg0)->getPointerElementType()->isStructTy()) {
                if (cast<PointerType>(Arg0)->getPointerElementType()->getStructName() == "struct.Tensor")
                    funcs.insert(fname);
            } else {
                Type *Arg1 = type->getParamType(1);
                if (Arg1->isPointerTy()) Arg1 = cast<PointerType>(Arg1)->getPointerElementType();
                if (Arg1->isStructTy() && Arg1->getStructName() == "struct.Tensor")
                    funcs.insert(fname);
            }
        }
    }
}

bool WrapperPass::addWrapper(Function &F) {
    bool ret = false;
    if (F.getName() == "main") setDevice(F);
    for (auto it = F.begin(), ie = F.end(); it != ie; ++it) {
        BasicBlock &bb = *it;
        for (auto bi = bb.begin(), be = bb.end(); bi != be; ) {
            Instruction *inst = &*bi;
            CallInst *call = dyn_cast<CallInst>(inst);
            if (!call || !call->getCalledFunction()) {
                ++bi;
                continue;
            }
            StringRef callee = call->getCalledFunction()->getName();
            SmallString<64> wname;
            raw_svector_ostream stream(wname);
            stream << callee << "_wrapper";
            StringRef wrapper = stream.str();
            if (funcs.find(wrapper) == funcs.end()) {
                ++bi;
                continue;
            }
            Function *func = F.getParent()->getFunction(wrapper);
            SmallVector<Value *, 4> Args;
            int i = 0, n = call->getNumArgOperands();
            while (i < n) {
                if (!func->hasParamAttribute(i, Attribute::StructRet)) break;
                Args.push_back(call->getArgOperand(i++));
            }
            FunctionType *ftype = FunctionType::get(Type::getInt32Ty(F.getContext()), false);
            Constant *func1 = F.getParent()->getOrInsertFunction("get_device", ftype);
            CallInst *inst1 = CallInst::Create(func1, "", inst);
            Args.push_back(inst1);
            while (i < n) {
                Args.push_back(call->getArgOperand(i++));
            }
            CallInst *newcall = CallInst::Create(func, Args, "", inst);
            inst->replaceAllUsesWith(newcall);
            bi = inst->eraseFromParent();
            ret = true;
        }
    }
    return ret;
}

static Instruction *getInsertPos(Function &F) {
    BasicBlock &bb = F.getEntryBlock();
    Instruction *ret = bb.getTerminator();
    for (auto it = bb.begin(), ie = bb.end(); it != ie; ++it) {
        CallInst *inst = dyn_cast<CallInst>(&*it);
        if (!inst) continue;
        return inst;
    }
    return ret;
}

bool WrapperPass::setDevice(Function &F) {
    Instruction *insertP = getInsertPos(F);
    Value *argv = &*(F.arg_begin() + 1);
    SmallVector<Value *, 1> idx, idx1;
    idx.push_back(ConstantInt::get(Type::getInt64Ty(F.getContext()), 1));
    GetElementPtrInst *gep = GetElementPtrInst::CreateInBounds(argv, idx, "", insertP);
    LoadInst *ld = new LoadInst(gep, "", insertP);
    Constant *str = ConstantDataArray::getString(F.getContext(), StringRef("-cpu"));
    GlobalVariable *GV = new GlobalVariable(*(F.getParent()), str->getType(), true, GlobalValue::PrivateLinkage, str);
    idx1.push_back(ConstantInt::get(Type::getInt64Ty(F.getContext()), 0));
    idx1.push_back(ConstantInt::get(Type::getInt64Ty(F.getContext()), 0));
    GetElementPtrInst *s = GetElementPtrInst::CreateInBounds(GV, idx1, "", insertP);

    SmallVector<Type *, 2> argtys;
    argtys.push_back(Type::getInt8PtrTy(F.getContext()));
    argtys.push_back(Type::getInt8PtrTy(F.getContext()));
    FunctionType *fntype = FunctionType::get(Type::getInt32Ty(F.getContext()), argtys, false);
    Constant *func = F.getParent()->getOrInsertFunction("strcmp", fntype);
    SmallVector<Value *, 2> args;
    args.push_back(ld);
    args.push_back(s);
    CallInst *call = CallInst::Create(func, args, "", insertP);
    
    argtys.clear();
    argtys.push_back(Type::getInt32Ty(F.getContext()));
    fntype = FunctionType::get(Type::getVoidTy(F.getContext()), argtys, false);
    func = F.getParent()->getOrInsertFunction("set_device", fntype);
    args.clear();
    args.push_back(call);
    call = CallInst::Create(func, args, "", insertP);
    return true;
}

char WrapperPass::ID = 0;
static RegisterPass<WrapperPass> X("wrapper", "Wrapper pass");