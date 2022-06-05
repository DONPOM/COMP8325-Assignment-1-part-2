
      window.AmazonUIPageJS.when('p-detect').execute(function(d) {
        if(d.capabilities.touch) return;
        d.capabilities.touch = 'ontouchend' in document || navigator.MaxTouchPoints > 0 || navigator.msMaxTouchPoints > 0;
        if(d.capabilities.touch) document.documentElement.className += ' a-touch';
      });
    