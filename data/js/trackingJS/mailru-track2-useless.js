/*<![CDATA[*/
try {

timeInterval('renderbrowser');
rT.iGet = '';
for (var i = 0; i < rT.i.length; i++) {
        rT.comma = rT.iGet != '' ? ',' : '';
        var bP = rT.i[i].split('=');
        if (bP[0] != 'renderbrowser') {
                rT.iGet += (rT.comma + bP[0] + ':' + bP[1]);
        } else rT.renderbrowser = parseInt(bP[1]);
}
rT.comma = '';

rT.img = new Image(); rT.img.src='//mail.radar.imgsmail.ru/update?p=mail&t=main&v=' + rT.renderbrowser + '&i=' + rT.iGet + '&rnd=161353608' + (document.referrer ? '&r=' + escape(document.referrer) : '') + '';

} catch(e) {}
/*]]>*/