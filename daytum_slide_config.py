# `my_slide_config.py` 
c.TagRemovePreprocessor.remove_input_tags.add("hide")
c.SlidesExporter.reveal_theme="simple"
c.SlidesExporter.reveal_url_prefix="/"

# the following does the equivalent of --to slides and --post serve, see here: https://github.com/jupyter/nbconvert/blob/master/setup.py#L220 + https://github.com/jupyter/nbconvert/blob/master/nbconvert/nbconvertapp.py#L482 and here https://github.com/jupyter/nbconvert/blob/master/nbconvert/nbconvertapp.py#L238
app_settings = {
    "export_format": "slides"
}
c.NbConvertApp.update(app_settings)

# the following does the equivalent of --no-prompt, see here: https://github.com/jupyter/nbconvert/blob/master/nbconvert/nbconvertapp.py#L109-L114
exporter_settings = {
    'exclude_input_prompt' : True,
    'exclude_output_prompt' : True,
}
c.TemplateExporter.update(exporter_settings)
