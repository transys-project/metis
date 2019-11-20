(function() {
  var script = document.createElement('script');
  script.onload = function() {

    var stats = new MemoryStats();

    var elem = stats.domElement;
    elem.style.position = 'fixed';
    elem.style.right    = '0px';
    elem.style.bottom   = '0px';
    elem.style.zIndex   = 100000;

    document.body.appendChild( stats.domElement );

    requestAnimationFrame(function rAFloop(){
      stats.update();
      requestAnimationFrame(rAFloop);
    });
  };
  
  script.src = "http://127.0.0.1/memory-stats.js/memory-stats.js";
  document.head.appendChild(script);
})();
