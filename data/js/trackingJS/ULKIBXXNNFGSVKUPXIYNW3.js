(function () {
    var scheme = (("https:" == document.location.protocol) ? "https" : "http");
    var adnxs_domain = 'secure.adnxs.com';
    var aol_domain = 'secure.leadback.advertising.com';
    var rule = ["userhome_converted", "*userHome.php*"];
    var conversion = __adroll.get_conversion_value()
    if (conversion == null) {
        adroll_conversion_value = 0;
        adroll_currency = 'USD';
    } else if (conversion.currency == 'USC') {
        adroll_conversion_value = conversion.conv_value / 100
        adroll_currency = 'USD'
    }
    if (scheme == 'http') {
        adnxs_domain = 'ib.adnxs.com';
        aol_domain = 'leadback.advertising.com';
    }
    var el = document.createElement("div");
    el.style["width"] = "1px";
    el.style["height"] = "1px";
    el.style["display"] = "inline";
    var content = unescape('%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//d.adroll.com/cm/r/out%22/%3e%0a%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//d.adroll.com/cm/f/out%22/%3e%0a%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//d.adroll.com/cm/b/out%22/%3e%0a%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//d.adroll.com/cm/w/out%22/%3e%0a%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//d.adroll.com/cm/l/out%22/%3e%0a%3cimg%20height%3d%221%22%20width%3d%221%22%20alt%3d%22%22%20style%3d%22display%3anone%22%20src%3d%22https%3a//www.facebook.com/offsite_event.php%3fid%3d0%26value%3d%5bCONV_VALUE%5d%26currency%3d%5bCURRENCY%5d%22%20/%3e%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//www.googleadservices.com/pagead/conversion/1032613984/%3flabel%3d5lhRCKKFqAYQ4OCx7AM%26amp%3bguid%3dON%26amp%3bscript%3d0%26amp%3bord%3d%5bord%5d%22/%3e%0a%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//d.adroll.com/cm/g/out%3fgoogle_nid%3dadroll%22/%3e%0a%3cimg%20src%3d%22%5bprotocol%5d%3a//ad.yieldmanager.com/pixel%3fadv%3d330500%26code%3dULKIBXXNNFGSVKUPXIYNW3_c%26t%3d2%22%20width%3d%221%22%20height%3d%221%22%20/%3e%0a%3cimg%20src%3d%22%5bprotocol%5d%3a//ad.yieldmanager.com/pixel%3fadv%3d274138%26code%3dULKIBXXNNFGSVKUPXIYNW3_n%26t%3d2%22%20width%3d%221%22%20height%3d%221%22%20/%3e%0a%3cimg%20height%3d%221%22%20width%3d%221%22%20style%3d%22border-style%3anone%3b%22%20alt%3d%22%22%20src%3d%22%5bprotocol%5d%3a//www.googleadservices.com/pagead/conversion/1032613984/%3flabel%3dk6xuCIrfxwEQ4OCx7AM%26amp%3bguid%3dON%26amp%3bscript%3d0%26amp%3bord%3d%5bord%5d%22/%3e%0d%0a%0a');


    try {
} catch (e) {}

    var r = Math.random() * 10000000000000000;
    content = content.replace(/\[ord\]/gi, r);
    content = content.replace(/\[protocol\]/gi, scheme);
    content = content.replace(/\[adnxs_domain\]/gi, adnxs_domain);
    content = content.replace(/\[aol_domain\]/gi, aol_domain);
    content = __adroll.replace_external_data(content);
    el.innerHTML = content;
    __adroll._head().appendChild(el);
    if (typeof __adroll.set_pixel_cookie != 'undefined') {
        __adroll.set_pixel_cookie(adroll_adv_id, adroll_pix_id, "ULKIBXXNNFGSVKUPXIYNW3");
    }
}());

