
    (function() { // protect global namespace
      var prefix = ('https:' == document.location.protocol) ? 'https://ssl' : 'http://www';
      var script = document.createElement('script');
      script.type = 'text/javascript';
      script.src = prefix + '.gstatic.com/feedback/api.js';
      document.body.appendChild(script);
    })();
  