import os

empty_template = """{sidebar_entry}
<head>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js"></script>
<link rel="stylesheet" href="/Users/krogager/coding/website/jkrogager.github.io/pynot/assets/css/main.css"/>
</head>

                                    <header class="main">
                                      <h1>PyNOT : {task_name}</h1>
                                    </header>

                                    <p class="warning">
                                    Nothing here yet! Stay tuned... :)
                                    </p>
"""

all_tasks = {
    'spex': [
        # ['bias', 'bias', 'bias.html'],
        # ['corr', 'corr', 'corr.html'],
        ['sflat', 'sflat', 'sflat.html'],
        ['identify', 'identify', 'identify.html'],
        ['response', 'response', 'response.html'],
        ['wave1d', 'wave1d', 'wave1d.html'],
        ['wave2d', 'wave2d', 'wave2d.html'],
        ['skysub', 'skysub', 'skysub.html'],
        ['crr', 'crr', 'crr.html'],
        ['flux1', 'flux1', 'flux1.html'],
        ['flux2', 'flux2', 'flux2.html'],
        ['extract', 'extract', 'extract.html'],
    ],

    'phot': [
        # ['bias', 'bias', 'bias.html'],
        # ['corr', 'corr', 'corr.html'],
        ['imflat', 'imflat', 'imflat.html'],
        ['imtrim', 'imtrim', 'imtrim.html'],
        ['imcombine', 'imcombine', 'imcombine.html'],
        ['fringe', 'fringe', 'fringe.html'],
        ['sep', 'sep', 'sep.html'],
        ['wcs', 'wcs', 'wcs.html'],
        ['autozp', 'autozp', 'autozp.html'],
        ['findnew', 'findnew', 'findnew.html'],
    ]
}

for base, tasks in all_tasks.items():
    if not os.path.exists(base):
        os.mkdir(base)

    for task_name, sidebar_entry, fname in tasks:
        output_fname = os.path.join(base, fname)
        elements = {'task_name': task_name,
                    'sidebar_entry': sidebar_entry,
                    }
        with open(output_fname, 'w') as output:
            output.write(empty_template.format(**elements))
