
(function(){var l = document.location.href, u;if (l.indexOf('#') !== -1 && l.indexOf('&_suid') !== -1) {u = l.split('#')[1];var pos = u.indexOf('&_suid');u = u.substring(0, pos);document.location.href = u;}})();

if (typeof console !== 'undefined') {
    console.log('head scripts (' + window.location + ') completed after ' + ((new Date()).getTime() - window['jsStartTime']) + 'ms');
}
