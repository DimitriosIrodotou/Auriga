main:	*.tex
	pdflatex main
	bibtex main
	pdflatex main
	# pdflatex main

mnras: *.tex figs/*.p??
	rm -rf mnras.tar.gz
	tar cvzf mnras.tar.gz Makefile *.tex *.bib *.cls *.bst figs/*.p??

arxiv: *.tex figs/*.p??
	rm -rf arxiv.tgz
	tar cvzf arxiv.tar.gz Makefile *.tex *.bib *.bbl *.cls *bst figs/*.p??

overleaf:
	rm -rf overleaf.z
	zip overleaf.z Makefile *.tex *.bbl figs/*.p??

clean:
	/bin/rm -rf *~ *.bbl *.blg *.dvi *.log *aux main.pdf
