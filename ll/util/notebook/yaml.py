from IPython.display import HTML, Javascript, display


class YamlEditor:
    def __init__(self, yaml_str):
        self.yaml_str = yaml_str

    def _ipython_display_(self):
        # Set up CodeMirror editor options
        element_id = f"yaml-editor-{id(self)}"

        # Generate HTML for the CodeMirror editor
        # <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.58.3/theme/material.min.css">
        html = f"""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.58.3/codemirror.min.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.58.3/addon/fold/foldgutter.min.css">

            <textarea id="{element_id}" name="code">{self.yaml_str}</textarea>
        """
        _ = display(HTML(html))

        _ = display(
            Javascript(
                """
            require.config({
                packages: [{
                    name: "codemirror",
                    location: "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.58.3",
                    main: "codemirror"
                }],
                map: {
                    "*": {
                        "codemirror/lib/codemirror": "codemirror",
                    }
                }
            });

            require([
                "codemirror",
                "codemirror/mode/yaml/yaml", // for yaml mode
                "codemirror/addon/fold/foldcode", // for code folding
                "codemirror/addon/fold/foldgutter", // for code folding
                "codemirror/addon/fold/brace-fold", // for code folding
                "codemirror/addon/fold/comment-fold", // for code folding
                "codemirror/addon/fold/indent-fold", // for code folding
            ], function(CodeMirror) {
                var editor = CodeMirror.fromTextArea(document.getElementById("%(element_id)s"), {
                    mode: "yaml",
                    lineNumbers: true,
                    readOnly: true,
                    foldGutter: true,
                    gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"],
                });
            });
        """
                % {"element_id": element_id}
            )
        )
