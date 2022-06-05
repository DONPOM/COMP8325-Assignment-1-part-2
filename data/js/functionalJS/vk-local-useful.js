
var vk = {
  ads_rotate_interval: 120000,
  al: parseInt('3') || 4,
  id: 0,
  intnat: '1' ? true : false,
  host: 'vk.com',
  lang: 3,
  rtl: parseInt('') || 0,
  version: 16583,
  stDomains: 3,
  zero: false,
  contlen: 7662,
  loginscheme: 'https',
  ip_h: '34cfe967adbd861886',
  vc_h: '629f828b691b4a052b217bc5e319b2d7',
  navPrefix: '/',
  dt: parseInt('0') || 0,
  fs: parseInt('11') || 11,
  ts: 1417590872,
  tz: 10800,
  pd: 0,
  pads: 1,
  time: [2014, 12, 3, 10, 14, 32]
}

window.locDomain = vk.host.match(/[a-zA-Z]+\.[a-zA-Z]+\.?$/)[0];
var _ua = navigator.userAgent.toLowerCase();
if (/opera/i.test(_ua) || !/msie 6/i.test(_ua) || document.domain != locDomain) document.domain = locDomain;
var ___htest = (location.toString().match(/#(.*)/) || {})[1] || '', ___to;
if (vk.al != 1 && ___htest.length && ___htest.substr(0, 1) == vk.navPrefix) {
  if (vk.al != 3 || vk.navPrefix != '!') {
    ___to = ___htest.replace(/^(\/|!)/, '');
    if (___to.match(/^([^\?]*\.php|login)([^a-z0-9\.]|$)/)) ___to = '';
    location.replace(location.protocol + '//' + location.host + '/' + ___to);
  }
}

var StaticFiles = {
  'common.js' : {v: 1111},
  'common.css': {v: 485},
  'ie6.css'   : {v: 26},
  'ie7.css'   : {v: 18}
  ,'lang3_0.js':{v:3362},'index.css':{v:27},'index.js':{v:32},'ui_controls.css':{v:56},'ui_controls.js':{v:169}
}
