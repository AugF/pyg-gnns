all: gnn_survey.html
gnn_survey.html: *.md *.bib
	pandoc -s --katex  -F pandoc-crossref -F pandoc-citeproc -o gnn_survey.html gnn-performance-bottleneck-analysis.md
