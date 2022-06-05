
/*global googletag */
(function(){
    var immersionCompliment = false;
    
    if ((!immersion_adcode && !immersionCompliment)  || (immersion_adcode && immersionCompliment)) {
        document.write('<div id="div-gpt-ad-1392147725721-0" ></div>');
        googletag.cmd.push(function() {
            googletag.display('div-gpt-ad-1392147725721-0');
        });
    }
})();