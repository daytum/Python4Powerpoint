#!/usr/bin/env python

from nbformat import current as nbf

nb = nbf.new_notebook()
cells = []
cell = nbf.new_code_cell(input="""
import type_curve_monthly
from pptx import Presentation
from pptx.util import Inches
from datetime import date
from pd2ppt import df_to_table
import matplotlib.pyplot as plt

from IPython.display import Markdown
""", metadata={
    "slideshow": {
        "slide_type": "skip"
    }
})
cells.append(cell)
cells.append(nbf.new_text_cell('markdown', "# Daily Report", metadata={
    "slideshow": {
        "slide_type": "slide"
    }
}))
cells.append(nbf.new_code_cell('Markdown("## Generated on {:%m-%d-%Y}".format(date.today()))', metadata={
    "tags": [
        "hide"
    ]
}))

apis =  ['33025033410000', '33025021910000', '33025032430000', '33025031400000', 
         '33025024720000', '33025014750000', '33025027960000', '33025021060000',
         '33025021030000', '33025014120000', '33025017320000']
for api in apis:
    cells.append(nbf.new_text_cell('markdown', "# API = {}".format(api), metadata={
    "slideshow": {
        "slide_type": "slide"
    }
    }))
    cells.append(nbf.new_code_cell('p = type_curve_monthly.type_curve_plot(api="{}")'.format(api), metadata={
    "tags": [
        "hide"
    ]
    }))
    cells.append(nbf.new_code_cell('type_curve_monthly.type_curve_summary_from_db(api="{}")'.format(api), metadata={
    "tags": [
        "hide"
    ]
    }))
    
    


nb['worksheets'].append(nbf.new_worksheet(cells=cells))
nbf.convert(nb, to_version=4)

with open('report.ipynb', 'w') as f:
        nbf.write(nb, f, 'ipynb')