from pdflatex import PDFLaTeX
import os

latex_file = '/Users/venkateshmurugadas/Documents/nli_coliee/nli_legal_test_coliee_ovgu_local/attention_visuals/'
import os
files = []
for i in os.listdir(latex_file):
    if i.endswith('.tex'):
        files.append(i)

for file in files:

    pdfl = PDFLaTeX.from_texfile(latex_file+file)
    pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file= True)
