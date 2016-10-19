#ifndef READCONFIG_H_
#define READCONFIG_H_

#include <libconfig.h++>
#include "definitions.h"

int ConfigLookupInt(const char *config_string, const libconfig::Config &cfg, const bool default_config, int *val);
int ConfigLookupDouble(const char *config_string, const libconfig::Config &cfg, const bool default_config, double *val);
int ConfigLookupBool(const char *config_string, const libconfig::Config &cfg, const bool default_config, bool *val);
int ConfigLookupCString(const char *config_string, const libconfig::Config &cfg, const bool default_config, char **val);
int ReadConfig(const char *filename, const bool default_config, Param *params);

/*
int ReadSynthesisList(PARAM *params);
int Assign_config_parameters(const char *filename, struct config_t *conf, PARAM *params, int conf_type, int program_type);
int Check_parameter_validity(PARAM *params);
int Read_DNN_params(PARAM *params);
int Read_DNN_weights(PARAM *params, gsl_matrix **DNN_W);
int Read_input_minmax(PARAM *params, gsl_vector **input_minmax);
int Check_DNN_param_orders(PARAM *params);
*/

#endif /* READCONFIG_H_ */
