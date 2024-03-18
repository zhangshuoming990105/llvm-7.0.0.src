#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <memory>
#include <set>
#include <iostream>
#include <fstream>

using namespace clang;
using namespace std;

// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor>
{
public:
    MyASTVisitor(Rewriter &R, std::set<std::string> &F, std::string H, std::string S)
        : TheRewriter(R), Funcs(F), Header(H), Source(S) {}

    bool VisitDecl(Decl *D) {
        static std::ofstream outH(Header, ios::app);
        static std::ofstream outS(Source, ios::app);

        FunctionDecl *fd = dyn_cast<FunctionDecl>(D);
        if (fd && Funcs.find(fd->getName().str()) != Funcs.end()) {
            string ret = TheRewriter.getRewrittenText(fd->getReturnTypeSourceRange());
            string fname = fd->getName().str();
            //string newfunc = TheRewriter.getRewrittenText(fd->getReturnTypeSourceRange());
            //newfunc += " " + fd->getName().str() + "_wrapper(int device";
            
            string args;
            string paras;
            for (unsigned i = 0; i < fd->getNumParams(); ++i) {
                if (i != 0) {
                    paras += ", ";
                    args += ", ";
                }
                paras += TheRewriter.getRewrittenText(fd->getParamDecl(i)->getTypeSourceInfo()->getTypeLoc().getSourceRange()) + " Arg";
                paras += (char)('0'+i);
                args += "Arg";
                args += (char)('0'+i);
            }
            
            outH << ret << " " << fname << "_cpu(" << paras << ");" << std::endl;
            outH << ret << " " << fname << "_fpga(" << paras << ");" << std::endl;
            outH << ret << " " << fname << "_wrapper(int device, " << paras << ");" << std::endl;
            //newfunc += ")";
            //outH << newfunc << ";" << std::endl;
            /*outS << ret << " " << fname << "_wrapper(int device, " << paras << ") { ";
            outS << ret << " a = " << fname << "_fpga(" << args << "); ";
            outS << ret << " b = " << fname << "_cpu(" << args << "); ";
            if (ret == "Tensor") {
                outS << ret << " eq = EWSub_cpu(a, b); if (!eq.is_zero()) printf(\"" << fname << "wrong result\"); free_tensor(&eq);";
            } else {
                outS << ret << " eq = a - b; if (creal(eq) != 0 || cimag(eq) != 0) printf(\"" << fname << "wrong result\");";
            }
            outS << "return a;}" << std::endl;*/
            outS << ret << " " << fname << "_wrapper(int device, " << paras << ") { if (device) return " 
                    << fname << "_fpga(" << args << "); else return " << fname << "_cpu(" << args << ");}" << std::endl;
            /*newfunc += " { if (device) ";
            newfunc += fd->getName().str();
            newfunc += "_fpga(" + args + "); else ";
            newfunc += fd->getName().str();
            newfunc += "_cpu(" + args + "); }";
            outS << newfunc << std::endl;*/
        } 
        return false;
    }

private:
    Rewriter &TheRewriter;
    std::set<std::string> Funcs;
    std::string Header;
    std::string Source;
};


// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class MyASTConsumer : public ASTConsumer
{
public:
    MyASTConsumer(Rewriter &R, std::set<std::string> &F, std::string H, std::string S)
        : Visitor(R, F, H, S) {}

    // Override the method that gets called for each parsed top-level
    // declaration.
    virtual bool HandleTopLevelDecl(DeclGroupRef DR) {
        for (DeclGroupRef::iterator b = DR.begin(), e = DR.end();
             b != e; ++b)
            // Traverse the declaration using our AST visitor.
            Visitor.TraverseDecl(*b);
        return true;
    }

private:
    MyASTVisitor Visitor;
};

void addinclude(char *fileIn, char *header, char *source) {
    std::ifstream in(fileIn);
    std::ofstream outH(header), outS(source);

    if (!in) llvm::errs() << "Unable to open file: " << fileIn << '\n';
    if (!outH) llvm::errs() << "Unable to open file: " << header << '\n';
    if (!outS) llvm::errs() << "Unable to open file: " << source << '\n';
    if (!in || !outH || !outS) {
        if (in) in.close();
        if (outH) outH.close();
        if (outS) outS.close();
        return;
    }

    outH << "#include \"" << fileIn << "\"" << std::endl;
    outS << "#include \"" << header << "\"" << std::endl;

    in.close();
    outH.close();
    outS.close();
}

void getFuncs(std::set<string> &funcs, char *file) {
    ifstream in(file);
    if (!in) {
        llvm::errs() << "Unable to open file: " << file << '\n';
        return; 
    }
    
    std::string line;
    while (getline(in, line)) funcs.insert(line);
    in.close();
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        llvm::errs() << "Usage: emit-wrapper <inputheader> <wrapperfile> <outputheader> <outputsource>\n";
        return 1;
    }

    addinclude(argv[1], argv[3], argv[4]);

    std::set<string> funcs;
    getFuncs(funcs, argv[2]);
    
    // CompilerInstance will hold the instance of the Clang compiler for us,
    // managing the various objects needed to run the compiler.
    CompilerInstance TheCompInst;
    TheCompInst.createDiagnostics();
    TheCompInst.getDiagnostics().setSuppressAllDiagnostics();

    // Initialize target info with the default triple for our platform.
    auto TO = std::make_shared<TargetOptions>();
    TO->Triple = llvm::sys::getDefaultTargetTriple();
    TargetInfo *TI = TargetInfo::CreateTargetInfo(TheCompInst.getDiagnostics(), TO);
    TheCompInst.setTarget(TI);
    TheCompInst.createFileManager();
    FileManager &FileMgr = TheCompInst.getFileManager();
    TheCompInst.createSourceManager(FileMgr);
    SourceManager &SourceMgr = TheCompInst.getSourceManager();
    TheCompInst.createPreprocessor(TU_Complete);
    TheCompInst.createASTContext();
    TheCompInst.getPreprocessor().SetSuppressIncludeNotFoundError(true);

    // A Rewriter helps us manage the code rewriting task.
    Rewriter TheRewriter;
    TheRewriter.setSourceMgr(SourceMgr, TheCompInst.getLangOpts());

    // Set the main file handled by the source manager to the input file.
    const FileEntry *FileIn = FileMgr.getFile(argv[1]);
    FileID FID = SourceMgr.createFileID(FileIn, SourceLocation(), clang::SrcMgr::C_User);
    SourceMgr.setMainFileID(FID);

    // Create an AST consumer instance which is going to get called by
    // ParseAST.
    MyASTConsumer TheConsumer(TheRewriter, funcs, argv[3], argv[4]);

    // Parse the file to AST, registering our consumer as the AST consumer.
    ParseAST(TheCompInst.getPreprocessor(), &TheConsumer,
             TheCompInst.getASTContext());
    return 0;
}