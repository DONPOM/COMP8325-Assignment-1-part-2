window.GOVUK = window.GOVUK || {};

  GOVUK.Analytics = GOVUK.Analytics || {};
  var _gaq = _gaq || [];
  _gaq.push(['_setAccount', 'UA-26179049-1']);
  if(document.domain=='www.gov.uk') {
    _gaq.push(['_setDomainName', '.www.gov.uk']);
  } else {
    _gaq.push(['_setDomainName', document.domain]);
  }
  _gaq.push(['_setAllowLinker', true]);
    // track pixel density ratio
  if (window.devicePixelRatio) {
    _gaq.push(['_setCustomVar', 11, 'Pixel Ratio', String(window.devicePixelRatio), 2 ]);
  }
  // Search result placement tracking, set custom var and destroy the cookie
  if(GOVUK.cookie && GOVUK.cookie('ga_nextpage_params') !== null){
    var customVar = GOVUK.cookie('ga_nextpage_params').split(',');
    customVar[1] = parseInt(customVar[1], 10);
    customVar[4] = parseInt(customVar[4], 10);
    _gaq.push(customVar);
    GOVUK.cookie('ga_nextpage_params', null);
  }
_gaq.push(["_setCustomVar",2,"Format","homepage",3]);
GOVUK.Analytics.Format = "homepage";