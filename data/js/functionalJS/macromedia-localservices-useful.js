(function(window, document) {
  'use strict';

  var globalnavLocale = 'en_AU';
  var gnavImsFile = 'https://adobeid-na1' + adobe.dom.globalNavVars.ImsJS + '.services.adobe.com/ims/imslib.js?client_id=' + adobe.dom.globalNavVars.ImsClientId + '&amp;locale=' + globalnavLocale;
  var options = adobe.dom.adobeGlobalNavOptions;

  // Setting gnav locale option
  options.locale = globalnavLocale;

  // Setting gnav sections configured in page YFM
  options.sections = ['sitemap','search','profile'];

  
  options.progressiveEnhancement = true;
  

  // Setting gnav styles
  
  options.styles = options.styles || {};
  
  options.styles['buttonTextColor'] = 'white';
  
  

  // Adding custom listener to IMS onProfile event
  adobe.events.on('ims:onProfile', function(profile) {
    if(profile) {
      adobe.dom.profile();
    } else {
      adobe.dom.signOut();
    }
  });

  // Adding custom listener to gnav signIn event
  adobe.events.on('gnav:signIn', function(done) {
    done();
  });

  // Adding custom listener to gnav signOut event
  adobe.events.on('gnav:signOut', function(done) {
    adobe.dom.signOut();
    done();
  });

  // Creating gnav object
  window.adobeGlobalNav = new window.AdobeGlobalNav(options);

  document.writeln('\x3Cscript src="' + gnavImsFile + '">\x3C/script>');
})(window, document);