
      //Google Analytics
      var _gaq = _gaq || [];
      _gaq.push(['_setAccount', 'UA-29849174-1']);
      _gaq.push(['_setDomainName', 'mapquest.com']);
      _gaq.push(['_setAllowLinker', true]);
      _gaq.push(['_trackPageview']);
      _gaq.push(['galaxyTracker._setAccount','UA-40771409-1']); // additional tracker for all subdomains
      _gaq.push(['galaxyTracker._trackPageview']); 
    
      (function() {
        var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
        ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
        var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
      })();
