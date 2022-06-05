
// ads/afcAdUnit.ftl
$('.adsense-slot').each(function() {
    Abt.AdCode.addAdsenseAfcConfig($(this).attr('id'), Number($(this).attr('adsense-numlinks')), $(this).attr('adsense-displaylabel') === 'true');
});
// internal/utility.ftl

var abtGenericDomain='j';
var deviceType='PERSONAL_COMPUTER';
