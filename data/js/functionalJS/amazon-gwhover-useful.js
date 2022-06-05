
  P.when('SgHover').register('gw-hover-main', function(SgHover) {
    return new SgHover('#mainContent');
  });

  P.when('SgHoverHelper', 'gw-hover-main', 'gw-productdb')
   .register('gw-hover-main-helper', function(SgHoverHelper, gwHover, pdb) {
    return new SgHoverHelper(gwHover, pdb);
  });
