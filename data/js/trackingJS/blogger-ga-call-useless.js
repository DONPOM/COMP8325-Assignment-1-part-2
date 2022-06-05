
  
  var gaqPrefix = '';
  if (window.lastSetAccountCount) {
    gaqPrefix= 'b' + window.lastSetAccountCount + '.';
    window.lastSetAccountCount++;
  }
  var _gaq = _gaq || [];
  _gaq.push([gaqPrefix + '_setAccount', "UA-18003-7"]);
  _gaq.push([gaqPrefix + '_trackPageview' ]);
  
     _gaq.push([gaqPrefix + '_setAllowAnchor', true]);
  
  
    _gaq.push([gaqPrefix + '_setDetectClientInfo', false]);
  

  
    window.lastSetAccountCount = 1;
    (function() {
      var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
      ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
      (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(ga);
    })();

    function trackBloggerPageView(opt_url) {
      try {
        _gaq.push([gaqPrefix + '_trackPageview', opt_url]);
      } catch(err) {}
    }
  
