import glob
import os
from collections import defaultdict

class HTMLFormatError(Exception):
    pass



def get_title(text):
    """Find the <h1> tag in input text and retrieve its value"""
    all_lines = text.split('\n')
    for line in all_lines:
        if '<h1>' in line:
            line = line.replace('<h1>', '')
            line = line.replace('</h1>', '')
            title = line.strip()
            return title
    else:
        raise HTMLFormatError("could not find <h1> tag!")


def clean_page(text):
    """Clean the input text for sidebar entry name and <head> section"""
    all_lines = text.split('\n')
    # Remove the sidebar entry name in the first line
    all_lines.pop(0)
    if '</head>' in text:
        # Trim the <head>...</head> section
        for linenum, line in enumerate(all_lines, 1):
            if '</head>' in line:
                break
        all_lines = all_lines[linenum:]

    return '\n'.join(all_lines)



## TARGET                # ROOT_PATH     # TASK_PATH
#index.html              ./             ./tasks/
#tasks/spex/wave2d.html  ../../         ../../tasks/spex/test.html
def filter_html_files():
    pass

def build_file_tree(source_dir):
    tree = {}
    for root, dirs, fs in os.walk(source_dir):
        if '.DS_Store' in fs:
            fs.remove('.DS_Store')

        if root == source_dir:
            filelist = list(filter(lambda x: '.html' in x, fs))
            tree['./'] = sorted(filelist)
        else:
            if not source_dir.endswith('/'):
                root = root.replace(source_dir+'/', '')
            else:
                root = root.replace(source_dir, '')
            tree[root] = sorted([os.path.join(root, fname) for fname in fs if '.html' in fname])
    return tree


def get_root_path(fname):
    parts = fname.split('/')
    # Pop the file basename:
    _ = parts.pop(-1)
    if len(parts) == 0:
        return './'
    else:
        return '/'.join(['..' for _ in parts]) + '/'


def format_sidebar_link(link, title, root_path):
    return f'<li><a href="{root_path}{link}">{title}</a></li>'


def get_link_title(fname, source_dir='source'):
    with open(os.path.join(source_dir, fname)) as f:
        first_line = f.readline()

    display_field = first_line.strip()
    if display_field == '':
        raise ValueError("No sidebar name entry found in first line of %s!" % fname)
    return display_field



def build_sidebar(*, target, filetree, source_dir='source'):
    sidebar = defaultdict(list)
    root_path = get_root_path(target)
    
    # Initiate Sections and Links:
    for root, fnames in filetree.items():
        for fname in fnames:
            display = get_link_title(fname, source_dir)
            link = format_sidebar_link(fname, display, root_path)
            if 'spex' in root:
                sidebar['spex'].append(link)
            elif 'phot' in root:
                sidebar['phot'].append(link)
            # elif 'examples' in root:
            #     sidebar['ex'].append(link)
            else:
                sidebar['base'].append(link)
    
    # Format each section:
    html = []
    html.append('<ul>')
    indent = 4*' '
    for line in sidebar['base']:
        html.append(indent + line)

    # -- SPEX:
    html.append(indent + '<li>')
    html.append(indent + '<span class="opener">SPEX: tasks</span>')
    html.append(indent + '<ul>')
    for line in sidebar['spex']:
        html.append(2*indent + line)
    html.append(indent + '</ul>')
    html.append(indent + '</li>')

    # -- PHOT:
    html.append(indent + '<li>')
    html.append(indent + '<span class="opener">PHOT: tasks</span>')
    html.append(indent + '<ul>')
    for line in sidebar['phot']:
        html.append(2*indent + line)
    html.append(indent + '</ul>')
    html.append(indent + '</li>')

    # -- EXAMPLES:
    # html.append(indent + '<li>')
    # html.append(indent + '<span class="opener">Examples</span>')
    # html.append(indent + '<ul>')
    # for line in sidebar['ex']:
    #     html.append(2*indent + line)
    # html.append(indent + '</ul>')
    # html.append(indent + '</li>')

    html.append('</ul>')
    sidebar_indent = '                                '
    sidebar_html = '\n'.join([sidebar_indent + line for line in html])
    return sidebar_html



if __name__ == '__main__':
    source_dir = 'source'
    output_dir = 'html'
    template_filename = 'page_template'

    with open(template_filename) as temp_file:
        page_template = temp_file.read()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(" -  Created output directory:  %s" % output_dir)
    filetree = build_file_tree(source_dir)
    print(" -  Created file tree of source directory:  %s" % source_dir)

    for root, files in filetree.items():
        if root == './':
            pass
        else:
            output_base = os.path.join(output_dir, root)
            if not os.path.exists(output_base):
                os.mkdir(output_base)
                print(" -  Created output directory:  %s" % output_base)

        for fname in files:
            print(" -- Working on file: %s" % fname)
            root_path = get_root_path(fname)
            output_fname = os.path.join(output_dir, fname)
            abs_fname = os.path.join(source_dir, fname)
            with open(abs_fname) as task_file:
                html_section = task_file.read()
            
            sidebar_name = get_link_title(fname, source_dir)
            title = get_title(html_section)
            elements = dict()
            elements['title'] = title
            elements['section'] = clean_page(html_section)
            elements['root_path'] = root_path
            elements['sidebar'] = build_sidebar(target=fname,
                                                filetree=filetree,
                                                source_dir=source_dir)
            print(" -- Page title: <h1> %s </h1>" % title)
            print(" -- Sidebar entry: %s" % sidebar_name)
            #print(" -- Relative path to root: %s" % root_path)
    
            page_html = page_template.format(**elements)
            with open(output_fname, 'w') as output:
                output.write(page_html)
            print(" -- Saved page: %s" % output_fname)
            print("")
    
