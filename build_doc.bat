python -c "from sphinx.cmd.build import build_main;build_main(['-j2','-v','-T','-b','html','-d','docs/doctrees','docs/source','docs/build'])"
rem python -c "from sphinx.cmd.build import build_main;build_main(['-j2','-v','-T','-b','md','-d','docs/doctrees','docs/source','docs/buildmd'])"

