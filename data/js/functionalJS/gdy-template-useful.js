
selectstyle.addTemplate({
name: 'header currency',
mainUi: '<div class="selectstyleui"><div class="ss-value utility-bar-height" data-openit=\'{"addclass":{"selector":"#pch5 .utility-bar .select-currency","class":"oi-open"},"mask":"#pch5 .utility-bar .select-currency .selectstyleui .options-wrap","guide":".ss-options","animate":"height","speed":"200","group":"group1"}\'><span class="text-wrap pc-sprite-after currency-width"><span class="ss-var-abbreviation"></span></span><div class="down-arrow pc-sprite"></div></div><div class="options-wrap currency-options-wrap"><ul class="ss-options"></ul></div></div>',
optionUi: '<li class="ss-option"><span class="ss-content"></span><span class="ss-var-divider"></span></li>'
});
selectstyle.addTemplate({
name: 'header support',
mainUi: '<div class="selectstyleui"><div class="ss-value utility-bar-height" data-openit=\'{"addclass":{"selector":"#pch5 .utility-bar .select-support","class":"oi-open"},"mask":"#pch5 .utility-bar .select-support .selectstyleui .options-wrap","guide":".support-options-guide","animate":"height","speed":"200","group":"group1"}\'><span class="text-wrap pc-sprite-after"><span class="ss-var-hours"></span><span class="ss-var-phone"></span></span><div class="down-arrow pc-sprite"></div></div><div class="options-wrap support-options-wrap"><div class="support-options-guide"><div class="alt-locations">Alternate numbers</div><ul class="ss-options"></ul></div></div></div>',
optionUi: '<li class="ss-option"><div class="classy-row select-support-option"><div class="text-left"><span class="ss-var-hours"></span><span class="ss-var-phone"></span></div><div class="text-right"><a href="#" class="select-support-btn">Selected</a></div></div></li>'
});
selectstyle.addTemplate({
name: 'footer currency',
mainUi: '<div class="selectstyleui"><div class="ss-value" data-openit=\'{"mask":"#pcf .select-currency .selectstyleui .options-wrap","guide":".ss-options","animate":"height","speed":"200"}\'><span class="text-wrap pc-sprite-after"><span class="ss-var-abbreviation"></span></span><div class="down-arrow pc-sprite"></div></div><div class="options-wrap footer-currency-options-wrap"><ul class="ss-options"></ul></div></div>',
optionUi: '<li class="ss-option"><span class="ss-content"></span><span class="ss-var-divider"></span></li>'
});
/** Social Locale Fallbacks **/
if(typeof window['fosshare'] == 'undefined'){
window['fosshare'] = {};
}
window['fosshare']['facebookLocaleDefaults'] = {fallback:'en-us',en:'en-us',es:'es-la',de:'de-de'};
window['fosshare']['googleLocaleDefaults'] = {fallback:'en-us',en:'en-us',es:'es-419',zh:'zh-cn',pt:'pt-br'};
