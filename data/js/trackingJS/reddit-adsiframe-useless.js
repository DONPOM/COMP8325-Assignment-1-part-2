
  target = location.hash.substr(1);
  window.name = 'ad_main';
  var timeout = 0;
  var loadSponsorship = function() {
    timeout++;
    if(ados_results) {
      if(ados_results["sponsorship"]) {
        if (window.postMessage) {
          window.parent.postMessage('ados.createAdFrame:sponsorship', target);
        } else {
          iframe = document.createElement('iframe');
          iframe.src = 'http://www.reddit.com/static/createadframe.html';
          iframe.style.display = 'none';
          document.documentElement.appendChild(iframe);
        }
      }
    }
    else {
      if(timeout < 200) {
        setTimeout(loadSponsorship, 10);
      }
    }
  }
  $(window).load(loadSponsorship);
