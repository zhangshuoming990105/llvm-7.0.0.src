//===--- CGExprTensorC.cpp - Emit LLVM Code for C++ expressions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of TensorC expressions
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/IR/Value.h"

using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitTensorcMemberFunctionExpr(const CallExpr *E, ReturnValueSlot ReturnValue) {
  const MemberExpr *Member = dyn_cast<MemberExpr>(E->getCallee()->IgnoreImpCasts());
  const Expr *Base = Member->getBase();
  bool isarrow = Member->isArrow();
  QualType BaseType = Base->getType()->getCanonicalTypeInternal();
  if (isarrow) BaseType = Base->getType()->getAs<PointerType>()->getPointeeType()->getCanonicalTypeInternal();
  assert(isa<RecordType>(BaseType));
  assert(isa<FieldDecl>(Member->getMemberDecl()));
  const FieldDecl *Field = cast<FieldDecl>(Member->getMemberDecl());

  Address This = Address::invalid();
  if (isarrow)
    This = EmitPointerWithAlignment(Base);
  else
    This = EmitLValue(Base).getAddress();

  QualType FTy = getContext().getFieldFunctionType(BaseType, Field->getType());
  FTy = FTy.getCanonicalType();
  const FunctionType *FnType = cast<FunctionType>(FTy);

  CallArgList Args;
  Args.add(RValue::get(This.getPointer()), getContext().getPointerType(BaseType));
  EmitCallArgs(Args, dyn_cast<FunctionProtoType>(Field->getType().getTypePtr()), E->arguments());
  const CGFunctionInfo &FnInfo = CGM.getTypes().arrangeFreeFunctionCall(Args, FnType, false);
  llvm::Constant *Func = CGM.GetOrCreateFunctionForRecordField(cast<RecordType>(BaseType)->getDecl(), Field);
  CGCallee Callee = CGCallee::forDirect(Func);
  return EmitCall(FnInfo, Callee, ReturnValue, Args, nullptr, E->getExprLoc());
}

static StringRef getBinaryOperatorFuncName(BinaryOperatorKind Opc, int TensorScalar) {
  switch(Opc) {
    case BO_Add:
      return TensorScalar == 0 ? "EWAdd" : "EWAddScalar";
    case BO_Sub:
      return TensorScalar == 0 ? "EWSub" : (TensorScalar == 1 ? "EWSubScalar" : "EWAddScalar");
    case BO_Mul:
      return TensorScalar == 0 ? "EWMul" : "EWMulScalar";
    case BO_Div:
      return TensorScalar == 0 ? "EWDiv" : (TensorScalar == 1 ? "EWDivScalar" : "EWMulScalar");
    case BO_Rem:
      return TensorScalar == 0 ? "EWRem" : (TensorScalar == 1 ? "EWRemScalar" : "EWRemScalarTensor");
    case BO_Shl:
      return TensorScalar == 0 ? "EWShl" : (TensorScalar == 1 ? "EWShlScalar" : "EWShlScalarTensor");
    case BO_Shr:
      return TensorScalar == 0 ? "EWShr" : (TensorScalar == 1 ? "EWShrScalar" : "EWShrScalarTensor");
    case BO_LT:
      return TensorScalar == 0 ? "EWLT" : "EWLTScalar";
    case BO_LE:
      return TensorScalar == 0 ? "EWLE" : "EWLEScalar";
    case BO_GT:
      return TensorScalar == 0 ? "EWGT" : "EWGTScalar";
    case BO_GE:
      return TensorScalar == 0 ? "EWGE" : "EWGEScalar";
    case BO_EQ:
      return TensorScalar == 0 ? "EWEQ" : "EWEQScalar";
    case BO_NE:
      return TensorScalar == 0 ? "EWNE" : "EWNEScalar";
    default:
      llvm_unreachable("Unsupported binary operator");
      return "";
  }
}

RValue CodeGenFunction::EmitTensorcBinaryOperator(const BinaryOperator *E, ReturnValueSlot ReturnValue) {
  const Expr *LHS = E->getLHS(), *RHS = E->getRHS();
  QualType LHSType = LHS->getType(), RHSType = RHS->getType();
  const RecordType *LRecord = nullptr, *RRecord = nullptr;
  if (LHSType->isStructureType()) LRecord = LHSType->getAs<RecordType>();
  if (RHSType->isStructureType()) RRecord = RHSType->getAs<RecordType>();
  BinaryOperatorKind opc = E->getOpcode();
  bool istensor = (LRecord || RRecord) &&
		  	  	  (((LRecord && LRecord->getDecl()->getName() == "Tensor") || !LRecord) ||
				  ((RRecord && RRecord->getDecl()->getName() == "Tensor") || !RRecord));
  if (!istensor) {
    ErrorUnsupported(E, "aggregate binary expression");
    return RValue::getIgnored();
  }

  if (!LRecord && RRecord) {
    if ((opc >= BO_LT && opc <= BO_NE) ||
    		opc == BO_Add || opc == BO_Mul) {
      std::swap(LHS, RHS);
      std::swap(LHSType, RHSType);
      std::swap(LRecord, RRecord);

      if (opc >= BO_LT && opc <= BO_NE) {
        switch (opc) {
        case BO_LT:
          opc = BO_GT;
          break;
        case BO_GT:
          opc = BO_LT;
          break;
        case BO_LE:
          opc = BO_GE;
          break;
        case BO_GE:
          opc = BO_LE;
          break;
        default:
          break;
        }
      }
    }
  }

  QualType TensorTy = LHSType.getCanonicalType();
  if (!LRecord) TensorTy = RHSType.getCanonicalType();

  // Get function
  CallArgList Args;
  StringRef FnName;
  if (LRecord) {
    EmitCallArg(Args, LHS, LHSType);
    EmitCallArg(Args, RHS, RHSType);
    if (RRecord) 
      FnName = getBinaryOperatorFuncName(opc, 0);
    else
      FnName = getBinaryOperatorFuncName(opc, 1);
  } else {
    CallArgList Arg;
    EmitCallArg(Arg, RHS, RHSType);
    FunctionProtoType::ExtProtoInfo EPI;
    SmallVector<QualType, 2> ArgTys(2, TensorTy);
    QualType FnType = getContext().getFunctionType(TensorTy, ArgTys, EPI);
    const CGFunctionInfo &FnInfo =
        CGM.getTypes().arrangeFreeFunctionCall(Arg, cast<FunctionType>(FnType.getTypePtr()), false);
    llvm::FunctionType *FnTypeLLVM = CGM.getTypes().GetFunctionType(FnInfo);
    StringRef Fname;
    if (opc == BO_Sub) Fname = "EWNegative";
    else Fname = "EWReciprocal";
    llvm::Constant *Func = CGM.GetOrCreateFunction(FnTypeLLVM, Fname);
    CGCallee Callee = CGCallee::forDirect(Func);
    RValue r = EmitCall(FnInfo, Callee, ReturnValueSlot(), Arg, nullptr, E->getExprLoc());
    Args.push_back(CallArg(r, TensorTy));
    EmitCallArg(Args, LHS, LHSType);
    FnName = getBinaryOperatorFuncName(opc, -1);
  }

  FunctionProtoType::ExtProtoInfo EPI;
  SmallVector<QualType, 2> ArgTys(2, TensorTy);
  QualType FnType = getContext().getFunctionType(TensorTy, ArgTys, EPI);
  const CGFunctionInfo &FnInfo =
		  CGM.getTypes().arrangeFreeFunctionCall(Args, cast<FunctionType>(FnType.getTypePtr()), false);
  llvm::FunctionType *FnTypeLLVM = CGM.getTypes().GetFunctionType(FnInfo);
  llvm::Constant *Func = CGM.GetOrCreateFunction(FnTypeLLVM, FnName);
  CGCallee Callee = CGCallee::forDirect(Func);
  return EmitCall(FnInfo, Callee, ReturnValue, Args, nullptr, E->getExprLoc());
}
RValue CodeGenFunction::EmitTensorcUnaryOperator(const UnaryOperator *E, ReturnValueSlot ReturnValue) {
  const Expr *Val = E->getSubExpr();
  QualType Ty = Val->getType();

  // Emit arguments
  CallArgList Args;
  EmitCallArg(Args, Val, Ty);

  // Get llvm function
  FunctionProtoType::ExtProtoInfo EPI;
  SmallVector<QualType, 1> ArgTys(1, Ty);
  QualType FnType = getContext().getFunctionType(Ty, ArgTys, EPI);
  const CGFunctionInfo &FnInfo =
		  CGM.getTypes().arrangeFreeFunctionCall(Args, cast<FunctionType>(FnType.getTypePtr()), false);
  llvm::FunctionType *FnTypeLLVM = CGM.getTypes().GetFunctionType(FnInfo);
  llvm::Constant *Func = CGM.GetOrCreateFunction(FnTypeLLVM, "EWNegative");

  // Emit call
  CGCallee Callee = CGCallee::forDirect(Func);
  return EmitCall(FnInfo, Callee, ReturnValue, Args, nullptr, E->getExprLoc());
}
RValue CodeGenFunction::EmitTensorcBinAssign(const BinaryOperator *E, ReturnValueSlot ReturnValue) {
  QualType LHSType = E->getLHS()->getType(), RHSType = E->getRHS()->getType();

  // Emit arguments
  CallArgList Args;
  LValue LHS = EmitLValue(E->getLHS());
  Args.add(RValue::get(LHS.getPointer()), getContext().getPointerType(LHSType));
  EmitCallArg(Args, E->getRHS(), RHSType);

  // Get function
  StringRef FnName = "copy";
  if (RHSType->isArithmeticType()) FnName = "copy_scalar";

  FunctionProtoType::ExtProtoInfo EPI;
  SmallVector<QualType, 2> ArgTys;
  ArgTys.push_back(getContext().getPointerType(LHSType));
  ArgTys.push_back(RHSType);
  QualType FnType = getContext().getFunctionType(LHSType, ArgTys, EPI);
  const CGFunctionInfo &FnInfo =
		  CGM.getTypes().arrangeFreeFunctionCall(Args, cast<FunctionType>(FnType.getTypePtr()), false);
  llvm::FunctionType *FnTypeLLVM = CGM.getTypes().GetFunctionType(FnInfo);
  llvm::Constant *Func = CGM.GetOrCreateFunction(FnTypeLLVM, FnName);

  // Emit call
  CGCallee Callee = CGCallee::forDirect(Func);
  return EmitCall(FnInfo, Callee, ReturnValue, Args, nullptr, E->getExprLoc());
}
RValue CodeGenFunction::EmitTensorcTensorSliceExpr(const TensorSliceExpr *E, ReturnValueSlot ReturnValue) {
  const Expr *Base = E->getBase();
  QualType BaseTy = Base->getType();
  assert(BaseTy->isStructureType() && BaseTy->getAs<RecordType>()->getDecl()->getName() == "Tensor");

  QualType Ty = getContext().UnsignedIntTy;
  llvm::Type *LTy = ConvertTypeForMem(Ty);
  llvm::Value *Dim = llvm::ConstantInt::get(LTy, llvm::APInt(32, E->getDim()));

  // The lowerbound, upperbound, step must always be an integer, which is not an aggregate.
  // Emit lowerbound
  llvm::Value *This = EmitLValue(Base).getPointer();
  llvm::Value *LowerBound, *UpperBound, *Step;
  if (E->getLowerBound()) {
    LowerBound = EmitScalarExpr(E->getLowerBound());
    QualType LBTy  = E->getLowerBound()->getType();
    // Extend or truncate the index type to 32 or 64-bits.
    if (LowerBound->getType() != Int64Ty)
      LowerBound = Builder.CreateIntCast(LowerBound, Int64Ty, LBTy->isSignedIntegerOrEnumerationType(), "idxprom");
  } else {
	LowerBound = llvm::ConstantInt::get(Int64Ty, llvm::APInt(64, 0, true));
  }

  // Emit upperbound
  if (E->getUpperBound()) {
    UpperBound = EmitScalarExpr(E->getUpperBound());
    QualType UBTy = E->getUpperBound()->getType();
    // Extend or truncate the index type to 32 or 64-bits.
    if (UpperBound->getType() != Int64Ty)
      UpperBound = Builder.CreateIntCast(UpperBound, Int64Ty, UBTy->isSignedIntegerOrEnumerationType(), "idxprom");
  } else if (E->isRangeSlice()){
	CallArgList Args;
	SmallVector<QualType, 2> ArgTys;
	ArgTys.push_back(getContext().getPointerType(BaseTy));
	Args.add(RValue::get(This), ArgTys.back());
	ArgTys.push_back(Ty);
	Args.add(RValue::get(Dim), ArgTys.back());

	FunctionProtoType::ExtProtoInfo EPI;
	QualType FnType = getContext().getFunctionType(getContext().IntTy, ArgTys, EPI);
	const CGFunctionInfo &FnInfo =
			CGM.getTypes().arrangeFreeFunctionCall(Args, cast<FunctionType>(FnType.getTypePtr()), false);
	llvm::FunctionType *FnTypeLLVM = CGM.getTypes().GetFunctionType(FnInfo);
	llvm::Constant *Func = CGM.GetOrCreateFunction(FnTypeLLVM, "size");

	CGCallee Callee = CGCallee::forDirect(Func);
	UpperBound = EmitCall(FnInfo, Callee, ReturnValueSlot(), Args, nullptr, E->getExprLoc()).getScalarVal();
	UpperBound = Builder.CreateIntCast(UpperBound, Int64Ty, true, "idxprom");
  }

  // Emit step
  if (E->getStep()) {
    Step = EmitScalarExpr(E->getStep());
    QualType StepTy  = E->getStep()->getType();
    // Extend or truncate the index type to 32 or 64-bits.
    if (Step->getType() != Int64Ty)
      Step = Builder.CreateIntCast(Step, Int64Ty, StepTy->isSignedIntegerOrEnumerationType(), "idxprom");
  } else if (E->isRangeSlice()){
	Step = llvm::ConstantInt::get(Int64Ty, llvm::APInt(64, 1, true));
  }

  // Add args
  CallArgList Args;
  SmallVector<QualType, 5> ArgTys;
  ArgTys.push_back(getContext().getPointerType(BaseTy));
  Args.add(RValue::get(This), ArgTys.back());
  ArgTys.push_back(Ty);
  Args.add(RValue::get(Dim), ArgTys.back());
  ArgTys.push_back(getContext().LongLongTy);
  Args.add(RValue::get(LowerBound), ArgTys.back());
  if (E->isRangeSlice()) {
	ArgTys.push_back(getContext().LongLongTy);
    Args.add(RValue::get(Step), ArgTys.back());
    ArgTys.push_back(getContext().LongLongTy);
    Args.add(RValue::get(UpperBound), ArgTys.back());
  }

  // Get llvm function
  FunctionProtoType::ExtProtoInfo EPI;
  QualType FnType = getContext().getFunctionType(BaseTy, ArgTys, EPI);
  const CGFunctionInfo &FnInfo =
		  CGM.getTypes().arrangeFreeFunctionCall(Args, cast<FunctionType>(FnType.getTypePtr()), false);
  llvm::FunctionType *FnTypeLLVM = CGM.getTypes().GetFunctionType(FnInfo);
  llvm::Constant *Func = CGM.GetOrCreateFunction(FnTypeLLVM, E->isRangeSlice() ? "tensor_slice" : "tensor_select");

  // Emit call
  CGCallee Callee = CGCallee::forDirect(Func);
  return EmitCall(FnInfo, Callee, ReturnValue, Args, nullptr, E->getExprLoc());
}
