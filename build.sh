#!/usr/bin/env bash

# Build script for VisionAssist Senior Design Document
# Generates PDF from LaTeX source with proper error handling

set -e  # Exit on any error

echo "🔧 Building VisionAssist Senior Design Document..."
echo "=================================================="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install a LaTeX distribution."
    exit 1
fi

# Check if main.tex exists
if [ ! -f "main.tex" ]; then
    echo "❌ Error: main.tex not found in current directory."
    exit 1
fi

echo "📝 Running first pass (compilation)..."
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Error during first LaTeX compilation pass."
    echo "Check build.log for details:"
    tail -20 build.log
    exit 1
fi

echo "📝 Running second pass (cross-references)..."
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Error during second LaTeX compilation pass."
    echo "Check build.log for details:"
    tail -20 build.log
    exit 1
fi

# Check if PDF was generated
if [ -f "main.pdf" ]; then
    PDF_SIZE=$(ls -lh main.pdf | awk '{print $5}')
    echo "✅ Successfully generated main.pdf (${PDF_SIZE})"

    # Display PDF info if macOS or use file command
    if command -v mdls &> /dev/null; then
        PAGES=$(mdls -name kMDItemNumberOfPages main.pdf | awk -F '"' '{print $2}')
        echo "📄 Document: ${PAGES} pages"
    elif command -v pdfinfo &> /dev/null; then
        PAGES=$(pdfinfo main.pdf | awk '/Pages:/ {print $2}')
        echo "📄 Document: ${PAGES} pages"
    fi

    echo "📂 Output location: $(pwd)/main.pdf"
    echo ""
    echo "🎉 Build completed successfully!"
    echo "   PDF is ready for review and submission."

else
    echo "❌ Error: PDF file was not generated."
    echo "Check build.log for details."
    exit 1
fi

# Clean up auxiliary files if requested
if [ "$1" = "--clean" ]; then
    echo "🧹 Cleaning up auxiliary files..."
    rm -f *.aux *.log *.out *.toc *.bbl *.blg *.bcf *.run.xml
    echo "✅ Cleanup completed."
fi

echo "=================================================="
echo "Done! 🚀"