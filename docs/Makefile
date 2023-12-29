
.PHONY: html

html:
	python3 build_html.py

update: html
	rsync -rv html/* /Users/krogager/coding/jkrogager.github.io/pynot
 
clean:
	rm html/*.html html/tasks/*.html html/tasks/spex/*.html html/tasks/phot/*.html

