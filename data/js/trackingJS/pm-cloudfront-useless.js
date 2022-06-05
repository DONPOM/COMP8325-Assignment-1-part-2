
$j(window).load(function() {
    function IsImageOk(img) {
       return (!img.complete) ? false : !(typeof img.naturalWidth !="undefined" && img.naturalWidth == 0);
    }
    var root = "/";

    for (var i = 0; i < document.images.length; i++) {
        var src = document.images[i].src;
        if(src.indexOf('cloudfront.net')==-1 && src.indexOf('s3.amazonaws.com/tc-photos/')==-1) continue;
        if (!IsImageOk(document.images[i])) {
            var newstr = document.images[i].src.replace("http://dfmlo8oja8g1e.cloudfront.net/","");
            newstr = newstr.replace("http://d2oqjo3nc0aqra.cloudfront.net/","");
            newstr = newstr.replace("http://s3.amazonaws.com/tc-photos/","");

            var param = newstr.split('/');
            if (param.length != 4) {
                continue;
            }
            var imagesplit= param[3].split('.');
            var newurl=root+'photos/'+param[1]+"/"+param[2]+"/"+param[0]+"0S"+imagesplit[0]+"/alt/"+param[3];
            document.images[i].src = newurl;
        }
    }
});
