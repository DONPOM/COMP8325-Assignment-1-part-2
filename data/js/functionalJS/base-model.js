/**
 * The default base model for NYT5 Backbone components
 *
 * <p><b>Require Path:</b> foundation/model/base-model</p>
 *
 * @module Foundation
 * @class BaseModel
 * @constructor
 * @extends Backbone.Model
**/
define('foundation/models/base-model',[
    'jquery/nyt',
    'underscore/nyt',
    'backbone/nyt',
    'foundation/views/page-manager',
    'foundation/base-mixin'
], function ($, _, Backbone, pageManager, base) {
    'use strict';

    var BaseModel = Backbone.Model.extend(

        _.extend({}, base)
    );

    return BaseModel;
});
