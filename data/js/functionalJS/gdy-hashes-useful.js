
$(document).ready(function() {
var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&'),
qs = [],
date = new Date();
for(var i = 0; i < hashes.length; i++) {
var hash = hashes[i].split('=');
qs.push(hash[0]);
qs[hash[0]] = hash[1];
}
$('#shareme').fosShare({});
if($("#searchDomainName").val().length) {
$("#searchDomainName").val("");
}
$(".gdhp-search-box2").gdhpDomainSearch({
'domainInputId': '#searchDomainName2',
'tldInputId': '#searchTldName2',
'tldToggleId': '#searchTldToggle',
'tldListId': '#searchTldList',
'searchButton': '#domainSearchAction2',
'searchUrl': 'https://au.godaddy.com/domains/search.aspx?ci=54814',
'idnUrl': 'https://au.godaddy.com/domains/search.aspx?ci=82474',
'domainInputLabelText': 'Enter a <span class="b">domain name</span>',
'focusOnLoad' : false
});
$(".gdhp-search-box").gdhpDomainSearch({
'domainInputId': '#searchDomainName',
'tldInputId': '#searchTldName',
'tldToggleId': '#searchTldToggle',
'tldListId': '#searchTldList',
'searchButton': '#domainSearchAction',
'searchUrl': 'https://au.godaddy.com/domains/search.aspx?ci=54814',
'idnUrl': 'https://au.godaddy.com/domains/search.aspx?ci=82474',
'domainInputLabelText': 'Enter a <span class="b">domain name</span>'
});
$('.search-form').on('submit', function (e){
e.preventDefault();
$('button[name="searchButton"]', this).addClass('disabled');
var params = {
query:$(this).serialize(),
searchUrl:$(this).attr('data-url')
};
DomainSearch(params,function(err,NextStepUrl){
if(typeof NextStepUrl === 'undefined')
{
NextStepUrl = 'https://au.godaddy.com/domains/search-new.aspx';
}
if(err == null )
{
window.location = NextStepUrl;
}
else{
if(console != "undefined")
console.log(err);
}
});
});
function DomainSearch(params, cb) {
$.ajax({
url: params.searchUrl,
type: 'POST',
data: params.query,
error: function (XMLHttpRequest, textStatus, errorThrown) {
var data = { Error: errorThrown, Success: false };
cb(errorThrown,null);
},
success: function (data) {
cb(null,data.NextStepUrl);
}
});
}
$("#disclaimers > p > a").click(function () {
$("#g-modal").modal({
overlayId: "g-modal-overlay",
close: "false", // disables default close button
autoPosition: "true"
});
});
$(".g-modal-close").click(function () {
// Closes the last modal
$.modal.close();
});
});
$('.uk-cookie-close').click(function() {
$('.uk-cookie-banner').css("display","none");
var CookieDate = new Date;
CookieDate.setFullYear(CookieDate.getFullYear( ) +10);
document.cookie = 'uk_banner=closed; expires=' + CookieDate.toGMTString( ) + '; path=/';
})
if (document.cookie.indexOf("uk_banner") >= 0) {
$('.uk-cookie-banner').css("display","none");
} else {
$('.uk-cookie-banner').css("display","block");
}
if(navigator.userAgent.match(/iPad|iPhone|iPod/i)) {
$('head').append('<meta name="viewport" content="width=1150px">');
}
$('body').on('click','.js-gdhpclick',function(event){
var $clickedElement = $(event.currentTarget);
var $clickedUrl = $clickedElement.attr('data-url');
event.stopPropagation();
switch($clickedUrl)
{
case "video":
$('div#myVideoModal').modal('show');
break;
case "none":
break;
case "video-close":
$('div#myVideoModal').modal('hide');
break;
default:
window.location.href=$clickedUrl;
}
});
$(window).load(function () {
$('.bigtext').bigtext().css('visibility', 'visible');
});
