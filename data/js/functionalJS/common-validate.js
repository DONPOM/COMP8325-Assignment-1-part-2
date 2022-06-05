window.hasError = false;
window.errorContainer = "errorDiv";
function validateShow(objs, divId, returnValue, msg) {
    if (typeof(returnValue) != 'boolean') {
        alert('校验方法返回值不是boolean类型！');
        return;
    }
    if (!returnValue) {
        document.getElementById(divId).innerHTML = msg;
    }
    styleControl(objs, divId, returnValue);
    return returnValue;
}

function styleControl(objs, divId, isSuccess) {
    var cssTxt = 'border:2px solid #FFCC88;';
    var visible = 'block';
    if (isSuccess) {
        cssTxt = '';
        visible = 'none';
    }
    for (var i = 0; i < objs.length; i++) {
        var oldWidth = objs[i].style.width;
        objs[i].style.width = oldWidth;
        objs[i].focus();
    }
    document.getElementById(divId).style.display = visible;
}

function ipValidator(ip) {
    var ipRegexp = /^(25[0-5]|2[0-4][0-9]|[1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9])\.(25[0-5]|2[0-4][0-9]|[1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[0-9])$/;
    if (!ipRegexp.test(ip))
        return false;
    else 
        return true;
}
/*
 * function domainValidator(domain){ var ym
 * =/^[\w][\w-]*[\w]\.([\w][\w-]*[\w]\.)*(cn|us|uk|jp|hk|mo|com|edu|net|org|int|gov|mil|arpa|biz|info|name|pro|coop|aero|museum|cc|tv|asia|cat)$/ig;
 * var chinese=/[\u4e00-\u9fa5]/g ; var arrym=domain.split("."); var strym1; var
 * strym2; var strym3; var
 * zhongguo=/^[\w][\w-]*[\w]\.(ac|com|edu|net|org|gov|mil|bj|sh|tj|cq|he|sx|nm|ln|jl|hl|js|zj|ah|fj|jx|sd|ha|hb|hn|gd|gx|hi|sc|gz|yn|xz|sn|gs|qh|nx|xj|tw|hk|mo)\.(cn)$/ig;
 * var
 * national=/^[\w][\w-]*[\w]\.(int|com|edu|gov|mil|net|org|biz|info|pro|name|museum|coop|aero|xxx|idv|arts|shop|firm|nom|web|mobi|asia|me|travel|tv|cc|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cf|cg|ch|ci|ck|cl|cm|cn|co|cq|cr|cu|cv|cx|cy|cz|de|dj|dk|dm|do|dz|ec|ee|eg|eh|es|et|ev|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gh|gi|gl|gm|gn|gp|gr|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|in|io|iq|ir|is|it|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|mg|mh|ml|mm|mn|mo|mp|mq|mr|ms|mt|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nt|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|pt|pw|py|qa|re|ro|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|st|su|sy|sz|tc|td|tf|tg|th|tj|tk|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|va|vc|ve|vg|vn|vu|wf|ws|ye|yu|za|zm|zr|zw)$/ig;//国际顶级域名
 * if(arrym.length>=2){//域名至少两级 if(chinese.test(domain)){ //中文域名
 * //alert("中文域名"); return true; }else{//英文域名 //判断是否独立域名 if(arrym.length==3){
 * if(zhongguo.test(domain)){ //alert("中国域名正确"); return true; }else{
 * //alert("中国域名错误"); return false; } }else if(arrym.length==2){
 * if(national.test(domain)){ //alert("国际域名正确"); return true; }else{
 * //alert("国际域名错误"); return false; } } } }else{ //alert("域名错误"); return false; } }
 */
function domainValidator(str_url) {

    var strRegex = "^((https|http|ftp|rtsp|mms)?://)" 
    + "?(([0-9a-z_!~*'().&=+$%-]+: )?[0-9a-z_!~*'().&=+$%-]+@)?" // ftp的user@
    + "(([0-9]{1,3}\.){3}[0-9]{1,3}" // IP形式的URL- 199.194.52.184
    + "|" // 允许IP和DOMAIN（域名）
    + "([0-9a-z_!~*'()-]+\.)*" // 域名- www.
    + "([0-9a-z][0-9a-z-]{0,61})?[0-9a-z]\." // 二级域名
    + "[a-z]{2,6})" // first level domain- .com or .museum
    + "(:[0-9]{1,4})?" // 端口- :80
    + "((/?)|" // a slash isn't required if there is no file name
    + "(/[0-9a-z_!~*'().;?:@&=+$,%#-]+)+/?)$";
    var re = new RegExp(strRegex);
    if (re.test(str_url)) {
        return (true);
    } else {
        return (false);
    }
}

function spaceValidator(val) {
    if (val.replace(/(^\s*)|(\s*$)/g, "").length < val.length)
        return false
    return true;
}

// 显示错误信息 author:tangjm
function showErrorMessage(objArr , id , msg) {
    var errCtnObj = document.getElementById(id);
    errCtnObj.innerHTML = msg;
    for (var i = 0; i < objArr.length; i++) {
        var oldWid = objArr[i].style.width;
        objArr[i].style.cssText = "border:2px solid #FFCC88; width:" + oldWid + ";";
    }
    errCtnObj.style.display = 'block';
    window.location.href = '#top';
}
// 隐藏错误信息 author:tangjm
function hideErrorContainer(objArr, id) {
    for (var i = 0; i < objArr.length; i++) {
        var oldWid = objArr[i].style.width;
        objArr[i].style.cssText = "width:" + oldWid + ";";
    }
    document.getElementById(id).style.display = 'none';
}


/**
 * 校验域名格式
 * 
 * @param domain
 * @return
 */
function validateDomainName(domain) {

    var info = false;
    domain = domain.replace(/(^\s*)|(\s*$)/g, "");

    var strym = domain.replace(new RegExp("。", "g"), ".");

    var arrym = strym.split("\.");

    // 代表中文的正则
    var chinese = "([\u4e00-\u9fa5]+)";
    // 代表英文域名中最低级域名格式的正则
    var duli0 = "^(([a-z0-9][a-z0-9-]{0,61}[a-z0-9])|[a-z0-9]{1,63})$";
    // 代表英文3级域名的2级格式的正则
    var duli1 = "^(ac|com|edu|net|org|gov|mil|bj|sh|tj|cq|he|sx|nm|ln|jl|hl|js|zj|ah|fj|jx|sd|ha|hb|hn|gd|gx|hi|sc|gz|yn|xz|sn|gs|qh|nx|xj|tw|hk|mo)$";
    // 2级的顶级域名格式
    var duli2 = "^(int|com|edu|gov|mil|net|org|biz|info|pro|name|museum|coop|aero|xxx|idv|arts|shop|firm|nom|web|mobi|asia|me|travel|tv|cc|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cf|cg|ch|ci|ck|cl|cm|cn|co|cq|cr|cu|cv|cx|cy|cz|de|dj|dk|dm|do|dz|ec|ee|eg|eh|es|et|ev|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gh|gi|gl|gm|gn|gp|gr|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|in|io|iq|ir|is|it|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|mg|mh|ml|mm|mn|mo|mp|mq|mr|ms|mt|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nt|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|pt|pw|py|qa|re|ro|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|st|su|sy|sz|tc|td|tf|tg|th|tj|tk|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|va|vc|ve|vg|vn|vu|wf|ws|ye|yu|za|zm|zr|zw)$";

    // 中文域名下最低级域名的格式正则（包括2级域名的2级和3级域名的3级）
    var zhongwen0 = "^(([a-z0-9]|[\u4e00-\u9fa5])([a-z0-9-]|[\u4e00-\u9fa5]){0,20}([a-z0-9]|[\u4e00-\u9fa5])|[\u4e00-\u9fa5]{1,22})$";
    // 中文域名共分2级情况下顶级域名的格式正则
    var zhongwen1 = "^(\u4e2d\u56fd|\u4e2d\u570b|\u653f\u52a1|\u653f\u52d9|\u516c\u76ca|cn|com|hk|biz|tv|\u516c\u53f8|\u7f51\u7edc|\u7db2\u7d61|net|cc)$";
    // 中文域名共分3级情况下2级域名的格式正则
    var zhongwen2 = "^(\u653f\u52a1|\u653f\u52d9|\u516c\u76ca)$";
    // 中文域名共分3级情况下顶级域名的格式正则
    var zhongwen3 = "^(\u4e2d\u56fd|\u4e2d\u570b|cn)$";


    var reg_chinese = new RegExp(chinese, "i");
    var reg_duli0 = new RegExp(duli0, "i");
    var reg_duli1 = new RegExp(duli1, "i");
    var reg_duli2 = new RegExp(duli2, "i");
    var reg_zhongwen0 = new RegExp(zhongwen0, "i");
    var reg_zhongwen1 = new RegExp(zhongwen1, "i");
    var reg_zhongwen2 = new RegExp(zhongwen2, "i");
    var reg_zhongwen3 = new RegExp(zhongwen3, "i");

    if (arrym.length >= 2 && arrym.length <= 3) {

        // 判断是否是包含中文的域名
        if (null != reg_chinese.exec(strym)) {
            if (arrym.length == 2) {
                // 顶级域名符合格式，同时满足二级域名包含中文
                if (reg_zhongwen1.test(arrym[1]) && reg_chinese.exec(arrym[0]) && reg_zhongwen0.exec(arrym[0])) {
                    info = true;
                }
                // 域名分3级时
            } else {

                if (reg_zhongwen3.test(arrym[2]) && reg_zhongwen2.test(arrym[1]) && reg_chinese.exec(arrym[0]) && reg_zhongwen0.exec(arrym[0])) {
                    info = true;
                }
            }
            // 英文域名校验
        } else {
            if (domain.indexOf("。") == - 1) {

                var length = strym.length;
                if (length <= 67) {
                    for (var i = 0; i < arrym.length; i++) {
                        var strstr = arrym[i];
                        if (strstr.length > 63) {
                            info = false;
                            return info;
                        }
                    }
                    if (arrym.length == 3) {

                        if (reg_duli1.test(arrym[1]) && reg_duli0.test(arrym[0]) && arrym[2] == "cn") {

                            info = true;
                        }
                    } else if (arrym.length == 2) {
                        // 判断是否独立域名
                        if (reg_duli2.test(arrym[1]) && reg_duli0.test(arrym[0])) {
                            // 是独立域名
                            info = true;
                        }
                    }
                }
            }
        }
    }

    return info;
}

















