
    if (parent && parent != window && (browser.msie || browser.opera || browser.mozilla || browser.chrome || browser.safari || browser.iphone)) {
      document.getElementsByTagName('body')[0].innerHTML = '';
    } else {
      domReady();
      updateMoney(0);
gSearch.init();
if (window.qArr && qArr[5]) qArr[5] = [5, "by item", "", "goods", 0x00000100];
if (browser.iphone || browser.ipad || browser.ipod) {
  setStyle(bodyNode, {webkitTextSizeAdjust: 'none'});
}var qf = ge('quick_login_form'), ql = ge('quick_login'), qe = ge('quick_email'), qp = ge('quick_pass');
var qlb = ge('quick_login_button'), prgBtn = qlb;

var qinit = function() {
  setTimeout(function() {
    ql.insertBefore(ce('div', {innerHTML: '<iframe class="upload_frame" id="quick_login_frame" name="quick_login_frame"></iframe>'}), qf);
    qf.target = 'quick_login_frame';
  }, 1);
}

if (window.top && window.top != window) {
  window.onload = qinit;
} else {
  setTimeout(qinit, 0);
}

qf.onsubmit = function() {
  if (!ge('quick_login_frame')) return false;
  if (!trim(qe.value)) {
    notaBene(qe);
    return false;
  } else if (!trim(qp.value)) {
    notaBene(qp);
    return false;
  }
  lockButton(window.__qfBtn = prgBtn);
  prgBtn = qlb;
  clearTimeout(__qlTimer);
  __qlTimer = setTimeout(loginSubmitError, 30000);
  domFC(domPS(qf)).onload = function() {
    clearTimeout(__qlTimer);
    __qlTimer = setTimeout(loginSubmitError, 2500);
  }
  return true;
}

window.loginSubmitError = function() {
  showFastBox('Warning', 'Unable to complete encrypted authorization. This can happen if your date and time settings are not configured correctly on your system. Please check your date &amp; time settings and restart the browser.');
}
window.focusLoginInput = function() {
  scrollToTop(0);
  notaBene('quick_email');
}
window.changeQuickRegButton = function(noShow) {
  if (noShow) {
    hide('top_reg_link', 'quick_reg_button');
    show('top_search_link');
  } else {
    hide('top_search_link');
    show('top_reg_link', 'quick_reg_button');
  }
  toggle('top_switch_lang', noShow && window.langConfig && window.langConfig.id != 3);
}
window.submitQuickLoginForm = function(email, pass, opts) {
  setQuickLoginData(email, pass, opts);
  if (opts && opts.prg) prgBtn = opts.prg;
  if (qf.onsubmit()) qf.submit();
}
window.setQuickLoginData = function(email, pass, opts) {
  if (email !== undefined) ge('quick_email').value = email;
  if (pass !== undefined) ge('quick_pass').value = pass;
  var params = opts && opts.params || {};
  for (var i in params) {
    var el = ge('quick_login_' + i);
    if (el) {
      val(el, params[i]);
    } else {
      qf.appendChild(ce('input', {type: 'hidden', name: i, id: 'quick_login_' + i, value: params[i]}));
    }
  }
}

if (qlb) {
  qlb.onclick = function() { if (qf.onsubmit()) qf.submit(); };
}

if (browser.opera_mobile) show('quick_expire');

if (1) {
  hide('support_link_td');
}
var ts_input = ge('ts_input'), oldFF = browser.mozilla && parseInt(browser.version) < 8;
if (browser.mozilla && !oldFF) {
  setStyle(ts_input, {padding: (vk.rtl ? '3px 20px 6px 40px' : '3px 41px 6px 20px')});
}
placeholderSetup(ts_input, {back: false, reload: true});
if (browser.opera || browser.msie || browser.mozilla) {
  setStyle(ts_input, {padding: (vk.rtl ? '4px 20px 5px 40px' : '4px 41px 5px 20px')});
} else if (browser.chrome || browser.safari) {
  setStyle(ts_input, {padding: (vk.rtl ? '4px 21px 5px 40px' : '4px 40px 5px 21px')});
}

TopSearch.init();
if (browser.msie8 || browser.msie7) {
  var st = {border: '1px solid #a6b6c6'};
  if (hasClass(ge('ts_wrap'), 'vk')) {
    if (vk.rtl) st.left = '1px';
    else st.right = '0px';
  } else {
    if (vk.rtl) st.right = '146px';
    else st.left = '146px';
  }
  setStyle(ge('ts_cont_wrap'), st);
}
window.tsHintsEnabled = 1;
handlePageParams({"id":0,"pads":1,"loc":"","wrap_page":1,"width":791,"width_dec":160,"width_dec_footer":130,"body_class":"is_rtl font_default pads ","counters":"","pvbig":0,"pvdark":1});addEvent(document, 'click', onDocumentClick);Index.initNew();
JoinPhotoview.init([{"src":"\/images\/join\/prof_vk_1.png?5","width":790,"height":600,"desc":"Create a profile and share your news with friends and followers"},{"src":"\/images\/join\/news_vk_1.png?5","width":790,"height":600,"desc":"Stay in touch with your friends, relatives and celebrities"},{"src":"\/images\/join\/dial_vk_1.png?5","width":790,"height":600,"desc":"Chat with friends or groups of people via private messages"}]);
cur.lang = extend(cur.lang || {}, {
  index_screen_m_of_n: 'Screenshot {num} of {count}',
  index_to_main: 'Home',
  index_choose_sex: 'Select sex'
});
var sd = [[1,"Female"],[2,"Male "]];
if (sd) Index.initSexDD(sd);
placeholderSetup('ij_first_name', {back: true});
placeholderSetup('ij_last_name', {back: true});
var login = ge('quick_email');
if (login) {
  login.focus();
}cur.fbApp = '128749580520227';
cur.fbState = 'd58c8b120e5ae300d6';
Index.fbCheck(cur.fbApp, '');
var _tmr = _tmr || [];
_tmr.push({id: "2579437", type: "pageView", start: (new Date()).getTime()});
(function (d, w) {
var ts = d.createElement("script"); ts.type = "text/javascript"; ts.async = true;
ts.src = (d.location.protocol == "https:" ? "https:" : "http:") + "//top-fwz1.mail.ru/js/code.js";
var f = function () {var s = d.getElementsByTagName("script")[0]; s.parentNode.insertBefore(ts, s);};
if (w.opera == "[object Opera]") { d.addEventListener("DOMContentLoaded", f, false); } else { f(); }
})(document, window);
    }
  