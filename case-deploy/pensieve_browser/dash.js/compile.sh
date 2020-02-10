# filename=$1
grunt --config Gruntfile.js --force
cp ./dash.all.js ../video_server/
# cp ./dash.all.js ./compiled_code/$1_dash.all.js

browserify -p browserify-derequire -p bundle-collapser/plugin -o bundle.js -t [ babelify --presets [ @babel/preset-env  ] --plugins [ @babel/plugin-transform-class-properties ] ] script.js