/**
 * Creates an account mixin to minimize the duplication in account.
 *
 * <p><b>Require Path:</b> shared/account/views/account-mixin</p>
 *
 * @module Shared
 * @submodule Shared.Account
 * @namespace Account
 * @static AccountMixin
**/
define('foundation/helpers/model-mixin',[],function () {
    'use strict';

    var ModelMixin = {

        hasSynced: false,

        /**
         * Executes a callback function when the model is ready.
         *
         * @public
         * @method ready
         * @param callback {Function} A callback that is executed when the storage is ready.
         **/
        ready: function (callback) {
            if (callback && this.hasSynced) {
                callback();
            } else if (callback) {
                this.subscribeOnce(this.eventName, callback);
            }
        },

        /**
         * Checks if the data has been sync'd
         *
         * @public
         * @method isReady
         **/
        isReady: function () {
            return this.hasSynced;
        }
    };

    return ModelMixin;

});
