
// old globals... deprecate use, and switch to pc.page.sharedObjectMgr.set(key, value);
var pcj_pl_id = '1';
var pcj_idpredirect = "";
var pcj_ssoTargetKey = "target";
var pcj_isCart = false;
if(typeof pc != "undefined"){
pc.page.sharedObjectMgr.set('current_locale', 'en-au');
pc.page.sharedObjectMgr.set('pc_costco_supportnum', '877-818-3680');
pc.page.sharedObjectMgr.set('pc_default_support_num', '02 8023 8592');
pc.page.sharedObjectMgr.set('pc_default_support_hours', '24/7');
pc.page.sharedObjectMgr.set('pc_support_text', '');
pc.page.sharedObjectMgr.set('pc_qa_spoofable_pcsetdata', '');
pc.page.sharedObjectMgr.set('pc_market_id', 'en-au');
pc.page.sharedObjectMgr.set('pc_footer_skrill_visible', 'true');
pc.page.sharedObjectMgr.set('pc_skrill_valid_currencies', 'USD|GBP|EUR|CAD|AUD|CHF|INR|JPY');
pc.page.sharedObjectMgr.set('pc_args', '?ci=');
pc.page.sharedObjectMgr.set('pc_url_help','http://help.godaddy.com/');
pc.page.sharedObjectMgr.set('pc_url_bp','http://www.bobparsons.me/');
/** Enabled Social **/
pc.page.sharedObjectMgr.set('facebook-enabled',true);
pc.page.sharedObjectMgr.set('twitter-enabled',true);
pc.page.sharedObjectMgr.set('googleplus-enabled',false);
pc.page.sharedObjectMgr.set('pc_url_sales','https://au.godaddy.com/');
pc.page.sharedObjectMgr.set('pc_url_gui','https://gui.godaddy.com/');
}
if(typeof $pc == "undefined" && typeof jQuery != "undefined"){
var $pc = jQuery;
}
$pc(document).ready(function () {
if(typeof pc != "undefined"){
/** remove unsupported share locales **/
setTimeout(function(){
if(!pc.page.sharedObjectMgr.get('facebook-enabled')){
$('.fosshare .fosfbbox').css('display','none'); // fos share
$('ul.pcf-social-media-icon-list .pcf-social-media-icon-facebook').parent().css('display','none'); // pc footer
$('#fbbox.facebookDiv').css('display','none'); // old CDS1
}
if(!pc.page.sharedObjectMgr.get('twitter-enabled')){
$('.fosshare .fostwbox').css('display','none'); // fos share
$('ul.pcf-social-media-icon-list .pcf-social-media-icon-twitter').parent().css('display','none'); // pc footer
$('#twbox.twitterDiv').css('display','none'); // old CDS1
}
if(!pc.page.sharedObjectMgr.get('googleplus-enabled')){
$('.fosshare .fosgpbox').css('display','none'); // fos share
$('ul.pcf-social-media-icon-list .pcf-social-media-icon-google-plus').parent().css('display','none'); // pc footer
$('#gpbox.googlePlusDiv').css('display','none'); // old CDS1
}
// hide PC footer social title if no icons
var $pcFooterSocialLi = $('.pcf-social-media-icon-list li');
if($pcFooterSocialLi.length >= 1){
var visibleCount = 0;
$pcFooterSocialLi.each(function(){
if($(this).css('display') != 'none'){
visibleCount++;
}
});
if(visibleCount === 0){
$('.pcf-social-media-label').css('display','none');
}
}else{
$('.pcf-social-media-label').css('display','none');
}
},200);
/** Add locale to HTML tag **/
var currentLocale = pc.page.sharedObjectMgr.get('current_locale');
var htmlLang = $('html').attr('lang');
if(typeof htmlLang != 'undefined'){
if(htmlLang.toLowerCase() != currentLocale){
$('html').attr('lang',currentLocale);
}
}else{
$('html').attr('lang',currentLocale);
}
pcj_callext('pcj_setdata', 'https://gui.godaddy.com/pcjson/standardheaderfooter?ci=17368&callback=pcj_setdata');
var links = new pc.headers.c1.cdsdefault.links();
links.init('.sub-nav-menu > a:first-child,.qrylink > a:first-child, .header-tab > a:first-child, #submenu-mya-btn');
// todo: move these to the search init call
pcj_action_domain = 'https://au.godaddy.com/domains/search.aspx?ci=8962&checkAvail=1';
var menuItems = new pc.headers.c1.cdsdefault.menuItem();
menuItems.init();
var loginModalInit = "https://idp.godaddy.com/authentication/modalinit.aspx?version=7";
var loginModalContentUrl = "https://idp.godaddy.com/authentication/login.aspx?spkey=GDSWNET-130506072552001&version=7";
var loginModalBtn = new pc.headers.c1.cdsdefault.loginModal();
loginModalBtn.init("#pch5-login-btn", "81929", "1", loginModalInit, loginModalContentUrl, false);
var loginModalCreateAcct = new pc.headers.c1.cdsdefault.loginModal();
loginModalCreateAcct.init("#pct_createaccount", "81931", "1", loginModalInit, loginModalContentUrl, true);
var loginModalCreateAcctFooter = new pc.headers.c1.cdsdefault.loginModal();
loginModalCreateAcctFooter.init("#pct_createaccountfooter", "10530", "1", loginModalInit, loginModalContentUrl, true);
var footerAttributeLink = new pc.footers.c1.cdsdefault.footerAttributeLink();
footerAttributeLink.init();
var countryLanguageLink = new pc.footers.c1.cdsdefault.countryLanguageLink();
countryLanguageLink.init();
var countryModal = new linkTargetPopup();
countryModal.init("#currentCountryLabel","#pcf .countrySelectionPopup","960");
}
});
