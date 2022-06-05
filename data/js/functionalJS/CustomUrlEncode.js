/*
 * DPN 6/19/2014
 * Script to replace uri characters that asp.net considers invalid (<, >, *, %, :, & and ") with strings that it won't object to. Note that this script also replaces forward and back slashes, and so can't be used on an entire url string.
 */




/*
 * Returns the given string, uri-encoded, and with asp.net invalid characters (httpRuntime requestPathInvalidCharacters) replaced.
 */
function customUrlEncode(toEncode) {

    var encoded = toEncode;

    encoded = encoded.replace(/\&/ig, "_amp_");
    encoded = encoded.replace(/\*/ig, "_ast_");
    encoded = encoded.replace(/\</ig, "_lt_");
    encoded = encoded.replace(/\>/ig, "_gt_");
    encoded = encoded.replace(/\%/ig, "_pct_");
    encoded = encoded.replace(/\:/ig, "_cln_");
    encoded = encoded.replace(/\\/ig, "_bck_");
    encoded = encoded.replace(/\//ig, "_fwd_");
    encoded = encoded.replace(/\"/ig, "_quo_");

    return encodeURIComponent(encoded);
}


/*
 * Returns the given string, uri-decoded, and with replacement strings returned to their original characters.
 */
function customUrlDecode(toDecode) {

    var decoded = decodeURIComponent(toDecode);

    decoded = decoded.replace(/_amp_/ig, "&");
    decoded = decoded.replace(/_ast_/ig, "*");
    decoded = decoded.replace(/_lt_/ig, "<");
    decoded = decoded.replace(/_gt_/ig, ">");
    decoded = decoded.replace(/_pct_/ig, "%");
    decoded = decoded.replace(/_cln_/ig, ":");
    decoded = decoded.replace(/_bck_/ig, "\\");
    decoded = decoded.replace(/_fwd_/ig, "/");
    decoded = decoded.replace(/_quo_/ig, "\"");

    return decoded;
}
