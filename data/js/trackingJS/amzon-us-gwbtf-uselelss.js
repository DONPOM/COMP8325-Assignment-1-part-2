(function() {
    var avId = 'DfpVisibilityIdentifier_' + Math.floor(Math.random() * 1e16);
    document.write('\x3cdiv ' + 'class="GoogleActiveViewClass" ' + 'id="' + avId + '" ' + 'data-admeta-dfp="4215/amzn.us.gw.btf,300,250,17653192060730791213"\x3e');
    document.write('\x3c!-- aid: 27946454 --\x3e\n\x3c!-- cid: 40987861814 --\x3e\n\x3cscript\x3evar d16gCollapse\x3dtrue;frameElement.parentNode.style.display\x3d\x22none\x22;\x3c/script\x3e\n\n');
    document.write('\x3c/div\x3e');
    var avElement = document.getElementById(avId);
    if (avElement) {
        avElement['_avi_'] = 'BKlnrZedzVNCIGIet8AWE_IHgDQAAAAAQATgByAEJwAIC4AIA4AQBoAYfqBOAAQ';
        avElement['_eos_'] = false;
        avElement['_imm_'] = true;
        avElement['_cvu_'] = '';
        avElement['_adk_'] = 1468011375;
    }
    var glidar = document.createElement('script');
    glidar.type = 'text/javascript';
    glidar.async = true;
    glidar.src = '//pagead2.googlesyndication.com/pagead/js/lidar.js';
    var s = document.getElementsByTagName('script')[0];
    s.parentNode.insertBefore(glidar, s);
})();
