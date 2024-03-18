//===--- ExprTensorC.h - Classes for representing expressions ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Expr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPRTENSORC_H
#define LLVM_CLANG_AST_EXPRTENSORC_H

#include "clang/AST/Expr.h"

namespace clang {
/// \brief tensor slice
/// To specify an tensor slice in TensorC, array subscript
/// expressions are extended with the following syntax:
/// \code
/// [ index ]
/// [ lower-bound : ]
/// [ : ]
/// [ :: step ]
/// [ lower-bound :: step ]
/// [ :: ]
/// [ lower-bound ::]
/// [ lower-bound : upper-bound ]
/// [ : upper-bound ]
/// [ lower-bound : upper-bound : ]
/// [ : upper-bound : ]
/// [ lower-bound : upper-bound : step ]
/// [ : upper-bound : step ]
/// \endcode
/// Tensor slices are allowed on tensor. 
/// The lower-bound, upper-bound and step are integral type expressions.
/// The lower-bound and upper-bound are both included in range
/// The step must evaluate to non-negative integers. 
/// The lower-bound and upper-bound can be negative integers. When negative, it's equivalent to number + size of given tensor dimension
/// When the lower-bound is absent, it defaults to 0
/// When the upper-bound is absent, it defaults to the size of the given tensor dimension
/// When the step is absent, it defaults to 1
class TensorSliceExpr : public Expr {
  bool IsRangeSlice;
  unsigned Dim;
  enum { BASE, LOWER_BOUND, UPPER_BOUND, STEP, END_EXPR };
  Stmt *SubExprs[END_EXPR];
  SourceLocation LColonLoc, RColonLoc;
  SourceLocation RBracketLoc;

public:
  TensorSliceExpr(Expr *Base, Expr *LowerBound, Expr *UpperBound, Expr *Step, unsigned dim, bool isRange,
		          QualType Type, ExprValueKind VK, ExprObjectKind OK,
				  SourceLocation LColonLoc, SourceLocation RColonLoc, SourceLocation RBracketLoc)
      : Expr(
    		TensorSliceExprClass, Type, VK, OK,
            Base->isTypeDependent() ||
                (LowerBound && LowerBound->isTypeDependent()) ||
                (UpperBound && UpperBound->isTypeDependent()) ||
				(Step && Step->isTypeDependent()),
            Base->isValueDependent() ||
                (LowerBound && LowerBound->isValueDependent()) ||
				(UpperBound && UpperBound->isValueDependent()) ||
                (Step && Step->isValueDependent()),
            Base->isInstantiationDependent() ||
                (LowerBound && LowerBound->isInstantiationDependent()) ||
				(UpperBound && UpperBound->isInstantiationDependent()) ||
                (Step && Step->isInstantiationDependent()),
            Base->containsUnexpandedParameterPack() ||
                (LowerBound && LowerBound->containsUnexpandedParameterPack()) ||
				(UpperBound && UpperBound->containsUnexpandedParameterPack()) ||
                (Step && Step->containsUnexpandedParameterPack())),
			IsRangeSlice(isRange), Dim(dim), LColonLoc(LColonLoc), RColonLoc(RColonLoc), RBracketLoc(RBracketLoc) {
    SubExprs[BASE] = Base;
    SubExprs[LOWER_BOUND] = LowerBound;
    SubExprs[UPPER_BOUND] = UpperBound;
    SubExprs[STEP] = Step;
  }

  /// \brief Create an empty tensor slice expression.
  explicit TensorSliceExpr(EmptyShell Shell)
    : Expr(TensorSliceExprClass, Shell), IsRangeSlice(false), Dim(0) { }

  /// \brief Get dim of the tensor slice
  unsigned getDim() const { return Dim; }
  /// \brief Set dim of the tensor slice
  void setDim(unsigned dim) { Dim = dim; }

  /// \brief Check whether the subscript of tensor slice is range
  bool isRangeSlice() const { return IsRangeSlice; }
  /// \brief Set whether the subscript of tensor slice is range
  void setRangeSlice(bool isRange = true) { IsRangeSlice = isRange; }

  /// \brief Get base of the tensor slice.
  Expr *getBase() { return cast<Expr>(SubExprs[BASE]); }
  const Expr *getBase() const { return cast<Expr>(SubExprs[BASE]); }
  /// \brief Set base of the tensor slice.
  void setBase(Expr *E) { SubExprs[BASE] = E; }

  /// \brief Return original type of the base expression for tensor slice.
  static QualType getBaseOriginalType(const Expr *Base);

  /// \brief Get lower bound of tensor slice.
  Expr *getLowerBound() { return cast_or_null<Expr>(SubExprs[LOWER_BOUND]); }
  const Expr *getLowerBound() const {
    return cast_or_null<Expr>(SubExprs[LOWER_BOUND]);
  }
  /// \brief Set lower bound of the tensor slice.
  void setLowerBound(Expr *E) { SubExprs[LOWER_BOUND] = E; }

  /// \brief Get upper bound of tensor slice.
  Expr *getUpperBound() { return cast_or_null<Expr>(SubExprs[UPPER_BOUND]); }
  const Expr *getUpperBound() const {
    return cast_or_null<Expr>(SubExprs[UPPER_BOUND]);
  }
  /// \brief Set upper bound of the tensor slice.
  void setUpperBound(Expr *E) { SubExprs[UPPER_BOUND] = E; }

  /// \brief Get step of tensor slice.
  Expr *getStep() { return cast_or_null<Expr>(SubExprs[STEP]); }
  const Expr *getStep() const { return cast_or_null<Expr>(SubExprs[STEP]); }
  /// \brief Set step of the tensor slice.
  void setStep(Expr *E) { SubExprs[STEP] = E; }

  SourceLocation getLocStart() const LLVM_READONLY {
    return getBase()->getLocStart();
  }
  SourceLocation getLocEnd() const LLVM_READONLY { return RBracketLoc; }

  SourceLocation getLColonLoc() const { return LColonLoc; }
  void setLColonLoc(SourceLocation L) { LColonLoc = L; }

  SourceLocation getRColonLoc() const { return RColonLoc; }
  void setRColonLoc(SourceLocation L) { RColonLoc = L; }

  SourceLocation getRBracketLoc() const { return RBracketLoc; }
  void setRBracketLoc(SourceLocation L) { RBracketLoc = L; }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getBase()->getExprLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == TensorSliceExprClass;
  }

  child_range children() {
    return child_range(&SubExprs[BASE], &SubExprs[END_EXPR]);
  }
};
} // end namespace clang

#endif
