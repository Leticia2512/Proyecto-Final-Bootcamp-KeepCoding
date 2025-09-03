#!/bin/bash
jupyter nbconvert --to markdown EdaPDF.ipynb
sed -i.bak 's/^[ \t]*//' EdaPDF.md
cat header.yaml > EdaPDF_clean.md
sed 's/^#/\n\\newpage\n#/' EdaPDF.md >> EdaPDF_clean.md
pandoc EdaPDF_clean.md -o Eda.pdf --lua-filter=remove-code.lua --pdf-engine=xelatex 
