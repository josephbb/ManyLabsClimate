import pymc as pm
import pandas as pd
import numpy as np
from src.distributions import ZOIBProportion
from src.distributions import UpperInflatedTruncatedGeom
import pytensor.tensor as at


def transform_y_for_gpu_ZOIB(y):
    y = np.round(y, 2)
    y[y == 1.0] = 0.99999
    y[y == 0.0] = 0.00001
    return y


def get_policy_model(df):
    df["UniqueID"] = np.arange(df.shape[0])
    df_policy = pd.melt(
        df.loc[
            :,
            [
                "UniqueID",
                "Country",
                "condName",
                "BeliefADJ",
                "Policy1",
                "Policy2",
                "Policy3",
                "Policy4",
                "Policy5",
                "Policy6",
                "Policy7",
                "Policy8",
                "Policy9",
            ],
        ],
        id_vars=["UniqueID", "Country", "condName", "BeliefADJ"],
        var_name="Item",
        value_name="Policy",
    )

    print(df_policy.shape)

    df_policy = df_policy.dropna(axis=0)
    df_policy["Policy"].replace({"sesenta": 60.0}, inplace=True)
    df_policy["Policy"] = df_policy["Policy"].astype("float")

    # This function is necessary for GPU compatibility
    # as the full logp is calculated on the GPU
    # and values that are 1 and 0 are evaluated as -inf
    # causing sampling to fail.
    def transform_y_for_gpu_ZOIB(y):
        y = np.round(y, 2)
        y[y == 1.0] = 0.99999
        y[y == 0.0] = 0.00001
        return y

    # Factorize the categorical variables
    country_idxs, countries = pd.factorize(df_policy.Country)
    treatment_idxs, treatments = pd.factorize(df_policy.condName)
    participant_idxs, participants = pd.factorize(df_policy.UniqueID)
    item_idxs, items = pd.factorize(df_policy.Item)

    df_policy["country_idx"] = country_idxs
    df_policy["treatment_idxs"] = treatment_idxs

    # Particpant treatment and country variables
    p_t = df_policy.groupby([participant_idxs]).first()["treatment_idxs"].values
    p_c = df_policy.groupby([participant_idxs]).first()["country_idx"].values

    # Standardize belief
    df_policy["Belief"] = df_policy["BeliefADJ"]

    # Construct coordinates for the model
    coords = {
        "treatments": treatments,
        "countries": countries,
        "participants": participants,
        "obs_id": np.arange(len(country_idxs)),
        "items": items,
    }
    coords.update({"effect": ["intercept", "belief"]})

    with pm.Model(coords=coords) as model:
        # Mutable data for posterior predictive sampling
        country_idx = pm.MutableData("country_idx", country_idxs)
        treatment_idx = pm.MutableData("treatment_idx", treatment_idxs)
        participant_idx = pm.MutableData("participant_idx", participant_idxs)
        participant_country = pm.MutableData("participant_country", p_c)
        participant_treatment = pm.MutableData("participant_treatment", p_t)
        Belief = pm.MutableData("Belief", df_policy.Belief.values)
        country_predict = pm.MutableData("country_pred", [0])
        treatment_predict = pm.MutableData("treatment_pred", [0])
        bool_country = pm.MutableData("bool_country", [0])
        sim_belief = pm.MutableData("sim_belief", [0.0])
        y = pm.MutableData("y",  transform_y_for_gpu_ZOIB(df_policy.Policy.values / 100.0), dims="obs_id")
        u_σ_c = pm.Exponential.dist(lam=2)
        u_σ_t = pm.Exponential.dist(lam=2)

        # Obtain Cholesky factor for the covariance
        L_c, _, _ = pm.LKJCholeskyCov(
            "L_c", n=2, eta=4, sd_dist=u_σ_c, compute_corr=True, store_in_trace=False
        )

        L_t, _, _ = pm.LKJCholeskyCov(
            "L_t", n=2, eta=4, sd_dist=u_σ_t, compute_corr=True, store_in_trace=False
        )

        # # Parameters
        u_raw_c = pm.Normal("u_raw_c", mu=0, sigma=1, dims=("effect", "countries"))
        u_c = pm.Deterministic(
            "u_c", at.dot(L_c, u_raw_c).T, dims=("countries", "effect")
        )
        u_raw_t = pm.Normal("u_raw_t", mu=0, sigma=1, dims=("effect", "treatments"))
        u_t = pm.Deterministic(
            "u_t", at.dot(L_t, u_raw_t).T, dims=("treatments", "effect")
        )

        Alpha = pm.Normal("Alpha", 0, 1)
        Beta = pm.Normal("Beta", 0, 0.5)

        Item = pm.ZeroSumNormal("Item", 0.5, dims="items")

        participant_intercept = pm.Normal(
            "pi",
            u_c[participant_country, 0] + u_t[participant_treatment, 0] + Alpha,
            0.5,
            dims="participants",
        )
        participant_slope = pm.Normal(
            "ps",
            u_c[participant_country, 1] + u_t[participant_treatment, 1] + Beta,
            0.5,
            dims="participants",
        )

        mu = (
            participant_intercept[participant_idx]
            + participant_slope[participant_idx] * Belief
            + Item[item_idxs]
        )

        kappa = pm.Gamma("kappa", 7.5, 1)
        theta = pm.Gamma("theta", 1, 15)
        tau_offset = pm.Normal("tau_offset", 0, 1) 

        mu_pred = pm.Deterministic(
            "mu_pred",
            Alpha
            + bool_country * u_c[country_predict, 0]
            + u_t[treatment_predict, 0]
            + (
                u_c[country_predict, 1] * bool_country
                + u_t[treatment_predict, 1]
                + Beta
            )
            * sim_belief,
        )

        avg_pred = pm.Deterministic("avg_pred", theta * pm.invlogit(mu*tau_offset + mu) + \
                                                        (1-theta)* pm.invlogit(mu_pred)) 

        y_out = ZOIBProportion(
            "y_out",
            mu=pm.invlogit(mu),
            theta=theta,
            kappa=kappa,
            tau=pm.invlogit(mu*tau_offset + mu),
            observed=y,
            shape=Belief.shape,
        )
        return model, df_policy


def get_share_model(df, priors):

    df["UniqueID"] = np.arange(df.shape[0])
    df_SHARE = pd.melt(
        df.loc[:, ["UniqueID", "Country", "condName", "BeliefADJ", "SHAREcc"]],
        id_vars=[
            "UniqueID",
            "Country",
            "condName",
            "BeliefADJ",
        ],
        value_name="SHARE",
    )

    # Belief
    df_SHARE["Belief"] = df_SHARE["BeliefADJ"]

    # Drop empty observations
    df_SHARE = df_SHARE.dropna(axis=0)

    # Drop empty observations
    df_SHARE = df_SHARE.dropna(axis=0)

    temp = df_SHARE.copy()
    country_idxs, countries = pd.factorize(temp.Country)
    treatment_idxs, treatments = pd.factorize(temp.condName)

    temp = temp.dropna()
    temp["country_idx"] = country_idxs
    temp["treatment_idx"] = treatment_idxs

    coords = {
        "treatments": treatments,
        "countries": countries,
        "obs_id": np.arange(len(country_idxs)),
    }

    coords = {
        "treatments": treatments,
        "countries": countries,
        "obs_id": np.arange(len(country_idxs)),
    }

    coords.update({"effect": ["intercept", "slope"]})
    with pm.Model(coords=coords) as model_noncentered_share:

        # # Indexes for the data
        treatment_idx = pm.MutableData("treatment_idx", treatment_idxs, dims="obs_id")
        country_idx = pm.MutableData("country_idx", country_idxs, dims="obs_id")
        belief = pm.MutableData("belief", temp.Belief, dims="obs_id")

        country_sim = pm.MutableData("country_sim", [0])
        treatment_sim = pm.MutableData("treatment_sim", [0])
        bool_country = pm.MutableData("bool_country", [0])
        y = pm.MutableData("y", temp.SHARE)
        sim_belief = pm.MutableData("sim_belief", [0.0])

        u_σ_c = pm.Exponential.dist(lam=priors['lambda_val'])
        u_σ_t = pm.Exponential.dist(lam=priors['lambda_val'])

        # Obtain Cholesky factor for the covariance
        L_c, _, _ = pm.LKJCholeskyCov(
            "L_c", n=2, eta=2, sd_dist=u_σ_c, compute_corr=True, store_in_trace=False
        )

        L_t, _, _ = pm.LKJCholeskyCov(
            "L_t", n=2, eta=2, sd_dist=u_σ_t, compute_corr=True, store_in_trace=False
        )

        # # Parameters
        u_raw_c = pm.Normal("u_raw_c", mu=0, sigma=1, dims=("effect", "countries"))
        u_c = pm.Deterministic(
            "u_c", at.dot(L_c, u_raw_c).T, dims=("countries", "effect")
        )
        u_raw_t = pm.Normal("u_raw_t", mu=0, sigma=1, dims=("effect", "treatments"))
        u_t = pm.Deterministic(
            "u_t", at.dot(L_t, u_raw_t).T, dims=("treatments", "effect")
        )

        country_intercept_theta = pm.Deterministic(
            "country_intercept_theta", u_c[:, 0], dims="countries"
        )
        country_beta_theta = pm.Deterministic(
            "country_beta_theta", u_c[:, 1], dims="countries"
        )

        treatment_intercept_theta = pm.Deterministic(
            "treatment_intercept_theta", u_t[:, 0], dims="treatments"
        )
        treatment_beta_theta = pm.Deterministic(
            "treatment_beta_theta", u_t[:, 1], dims="treatments"
        )

        Alpha = pm.Normal("Alpha", 0, priors['alpha_sigma'])
        Beta = pm.Normal("Beta", 0, priors['beta_sigma'])

        mu2 = (
            Alpha
            + Beta * belief
            + country_intercept_theta[country_idx]
            + country_beta_theta[country_idx] * belief
            + treatment_intercept_theta[treatment_idx]
            + treatment_beta_theta[treatment_idx] * belief
        )

        sim_mu = pm.Deterministic(
            "sim_mu",
            Alpha
            + Beta * sim_belief
            + bool_country
            * (
                country_intercept_theta[country_sim]
                + country_beta_theta[country_sim] * sim_belief
            )
            + treatment_intercept_theta[treatment_sim]
            + treatment_beta_theta[treatment_sim] * sim_belief,
        )

        sim_avg = pm.Deterministic("sim_avg", pm.invlogit(sim_mu))

        theta = pm.Deterministic("theta", pm.invlogit(mu2))
        y_out = pm.Bernoulli("y_out", theta, observed=y)
    return model_noncentered_share, temp, df

def get_WEPT_model(df, priors):
    df["UniqueID"] = np.arange(df.shape[0])
    df_WEPT = pd.melt(
        df.loc[:, ["UniqueID", "Country", "condName", "BeliefADJ", "WEPTcc"]],
        id_vars=["UniqueID", "Country", "condName", "BeliefADJ"],
        value_name="WEPT",
    )

    # Scale Belief
    df_WEPT["Belief"] = df_WEPT["BeliefADJ"]

    # Drop empty observations
    df_WEPT = df_WEPT.dropna(axis=0)

    # Drop a few errant observations recording a 13
    df_WEPT = df_WEPT[~(df_WEPT["WEPT"] > 8)]
    df_WEPT = df_WEPT[~(df_WEPT["WEPT"] < 0)]

    # Add one for distributional considerations (below)
    df_WEPT["WEPT"] = df_WEPT["WEPT"] + 1

    temp = df_WEPT.copy()
    country_idxs, countries = pd.factorize(temp.Country)
    treatment_idxs, treatments = pd.factorize(temp.condName)

    temp["country_idx"] = country_idxs
    temp["treatment_idx"] = treatment_idxs

    coords = {
        "treatments": treatments,
        "countries": countries,
        "obs_id": np.arange(len(country_idxs)),
    }
    temp.shape
    y = temp["WEPT"]

    coords = {
        "treatments": treatments,
        "countries": countries,
        "obs_id": np.arange(len(country_idxs)),
    }

    coords.update(
        {
            "effect": [
                "intercept_geom",
                "slope_geom",
                "intercept_inflated",
                "slope_inflated",
            ]
        }
    )
    with pm.Model(coords=coords) as model_noncentered:
        upper = pm.ConstantData("upper", temp["WEPT"].max())

        # # Indexes for the data
        treatment_idx = pm.ConstantData("treatment_idx", treatment_idxs, dims="obs_id")
        country_idx = pm.ConstantData("country_idx", country_idxs, dims="obs_id")
        belief = pm.ConstantData("belief", temp.Belief, dims="obs_id")

        country_sim = pm.MutableData("country_sim", [0])
        treatment_sim = pm.MutableData("treatment_sim", [0])
        bool_country = pm.MutableData("bool_country", [0])
        y = pm.MutableData("y", temp.WEPT)
        sim_belief = pm.MutableData("sim_belief", [0.0])

        u_σ_c = pm.Exponential.dist(lam=priors['lambda_val'])
        u_σ_t = pm.Exponential.dist(lam=priors['lambda_val'])

        # Obtain Cholesky factor for the covariance
        L_c, _, _ = pm.LKJCholeskyCov(
            "L_c", n=4, eta=2, sd_dist=u_σ_c, compute_corr=True, store_in_trace=False
        )

        L_t, _, _ = pm.LKJCholeskyCov(
            "L_t", n=4, eta=2, sd_dist=u_σ_t, compute_corr=True, store_in_trace=False
        )

        # # Parameters
        u_raw_c = pm.Normal("u_raw_c", mu=0, sigma=1, dims=("effect", "countries"))
        u_c = pm.Deterministic(
            "u_c", at.dot(L_c, u_raw_c).T, dims=("countries", "effect")
        )
        u_raw_t = pm.Normal("u_raw_t", mu=0, sigma=1, dims=("effect", "treatments"))
        u_t = pm.Deterministic(
            "u_t", at.dot(L_t, u_raw_t).T, dims=("treatments", "effect")
        )

        country_intercept_geom = pm.Deterministic(
            "country_intercept_geom", u_c[:, 0], dims="countries"
        )
        country_beta_geom = pm.Deterministic(
            "country_beta_geom", u_c[:, 1], dims="countries"
        )
        country_intercept_theta = pm.Deterministic(
            "country_intercept_theta", u_c[:, 2], dims="countries"
        )
        country_beta_theta = pm.Deterministic(
            "country_beta_theta", u_c[:, 3], dims="countries"
        )

        treatment_intercept_geom = pm.Deterministic(
            "treatment_intercept_geom", u_t[:, 0], dims="treatments"
        )
        treatment_beta_geom = pm.Deterministic(
            "treatment_beta_geom", u_t[:, 1], dims="treatments"
        )
        treatment_intercept_theta = pm.Deterministic(
            "treatment_intercept_theta", u_t[:, 2], dims="treatments"
        )
        treatment_beta_theta = pm.Deterministic(
            "treatment_beta_theta", u_t[:, 3], dims="treatments"
        )

        Intercept_geom = pm.Normal("Intercept_geom", -1, 1)
        Beta_geom = pm.Normal("Beta_geom", 0, priors['beta_geom_sigma'])

        Intercept_theta = pm.Normal("Intercept_theta", -1, 1)
        Beta_theta = pm.Normal("Beta_theta", 0, priors['beta_theta_sigma'])

        mu1 = (
            Intercept_geom
            + Beta_geom * belief
            + country_intercept_geom[country_idx]
            + country_beta_geom[country_idx] * belief
            + treatment_intercept_geom[treatment_idx]
            + treatment_beta_geom[treatment_idx] * belief
        )

        mu2 = (
            Intercept_theta
            + Beta_theta * belief
            + country_intercept_theta[country_idx]
            + country_beta_theta[country_idx] * belief
            + treatment_intercept_theta[treatment_idx]
            + treatment_beta_theta[treatment_idx] * belief
        )

        sim_mu1 = pm.Deterministic(
            "sim_mu1",
            Intercept_geom
            + Beta_geom * sim_belief
            + bool_country
            * (
                country_intercept_geom[country_sim]
                + country_beta_geom[country_sim] * sim_belief
            )
            + treatment_intercept_geom[treatment_sim]
            + treatment_beta_geom[treatment_sim] * sim_belief,
        )

        sim_mu2 = pm.Deterministic(
            "sim_mu2",
            Intercept_theta
            + Beta_theta * sim_belief
            + bool_country
            * (
                country_intercept_theta[country_sim]
                + country_beta_theta[country_sim] * sim_belief
            )
            + treatment_intercept_theta[treatment_sim]
            + treatment_beta_theta[treatment_sim] * sim_belief,
        )
     


        sim_theta = pm.Deterministic("sim_theta", pm.invlogit(sim_mu2))
        sim_p = pm.Deterministic("sim_p", 1 / (1 + pm.invlogit(sim_mu1) * (upper - 1)))

        sim_mu = pm.Deterministic(
            "sim_mu", (1 - sim_theta) * (1 / sim_p-1) + sim_theta * (upper - 1)
        )

        y_out = UpperInflatedTruncatedGeom(
            "y_out",
            1 / (1 + pm.invlogit(mu1) * (upper - 1)),
            pm.invlogit(mu2),
            upper,
            observed=y,
        )
    return model_noncentered, temp


def get_belief_model(df, priors):
    
    df['UniqueID'] = np.arange(df.shape[0])
    df_belief = pd.melt(
        df.loc[:,['UniqueID', 'Country','condName','Belief1', 'Belief2','Belief3','Belief4']],
        id_vars=['UniqueID', 'Country','condName'],
        var_name='Item',
        value_name='Belief'
    )           

    #Drop nans, copy
    df_belief = df_belief.dropna(axis=0) #Some nan values for Belief, condition. A small proportion of overall data
    temp = df_belief.copy()

    #Rename and sort such that control is the first treatment. 
    temp['condName'].replace({'Control':'aaControl'}, inplace=True)
    temp.sort_values('condName', inplace=True, ascending=False)

    #Indexes
    country_idxs, countries = pd.factorize(temp.Country, )
    treatment_idxs, treatments = pd.factorize(temp.condName)
    participant_idxs, participants = pd.factorize(temp.UniqueID)
    item_idxs, items = pd.factorize(temp.Item)


    #Particpant-level indexes for country
    temp["country_idx"] = country_idxs
    participant_country_idxs = (
        temp.groupby(participant_idxs)["country_idx"].first().values
    )

    #Coords for model 
    coords = {
        "treatments": treatments,
        "countries": countries,
        "items": items,
        "participants": participants,
        "obs_id": np.arange(len(country_idxs)),
        "participant_id": np.arange(len(participants)),
        "item_id": np.arange(len(item_idxs)),
        "participant_country_id": np.arange(len(participant_country_idxs)),
    }
    coords.update({'effect':['Intercept', 'Slope']})


    with pm.Model(coords=coords) as model:
        #Mutable data for posterior predictive simulation
        country_idx = pm.MutableData("country_idx", country_idxs)
        treatment_idx = pm.MutableData("treatment_idx", treatment_idxs)
        country_sim = pm.MutableData("country_sim", [0])
        treatment_sim = pm.MutableData("treatment_sim", [0])
        bool_country = pm.MutableData("bool_country",[0])
        y = pm.MutableData("y", transform_y_for_gpu_ZOIB(temp.Belief.values/100.0))
        participant_country_idx = pm.MutableData("participant_country_idx", participant_country_idxs, dims='participants')
        item_idx = pm.MutableData("item_idx", item_idxs, dims="obs_id")
        particpant_idx = pm.MutableData("participant_idx", participant_idxs, dims="obs_id")
        sim_belief = pm.MutableData("sim_belief", [0.0])

        
        #Prior for sigmas 
        sigma_item = pm.Exponential("sigma_item", lam=priors['lambda_val'])
        sigma_participant = pm.Exponential("sigma_participant", lam=priors['lambda_val'])


        #Cholesky prior for treatment effects
        u_σ_t = pm.Exponential.dist(lam=[priors['lambda_val_intercept'],
                                         priors['lambda_val_slope']])
        L_t, _, _ = pm.LKJCholeskyCov(
            "L_t", n=2, eta=1, sd_dist=u_σ_t, compute_corr=True, store_in_trace=False
        )    
        
        #Treatment effects non-centered parameterization 
        u_raw_t = pm.math.stack([[0,0],
                                pm.Normal("u_raw_0", mu=0, sigma=1,dims='effect'), 
                                pm.Normal("u_raw_1", mu=0, sigma=1,dims='effect'), 
                                pm.Normal("u_raw_2", mu=0, sigma=1,dims='effect'), 
                                pm.Normal("u_raw_3", mu=0, sigma=1,dims='effect'),
                                pm.Normal("u_raw_4", mu=0, sigma=1,dims='effect'),
                                pm.Normal("u_raw_5", mu=0, sigma=1,dims='effect'),
                                pm.Normal("u_raw_6", mu=0, sigma=1,dims='effect'),
                                pm.Normal("u_raw_7", mu=0, sigma=1,dims='effect'), 
                                pm.Normal("u_raw_8", mu=0, sigma=1,dims='effect'),
                                pm.Normal("u_raw_9", mu=0, sigma=1,dims='effect'),
                                pm.Normal("u_raw_10", mu=0, sigma=1,dims='effect')], axis=0)
        u_t = pm.Deterministic("u_t", at.dot(L_t, u_raw_t.T).T, dims=("treatments", "effect"))


        a = pm.Normal('a', 0, 1)
        country_raw = pm.ZeroSumNormal("country_raw",1, dims="countries")
        country = pm.Deterministic("country", a + country_raw, dims="countries")
        # simple effects for item, zero-sum overall 
        item = pm.ZeroSumNormal("item", sigma=sigma_item, dims="items")
        
        # Let's assume that items may lead to more consistent or more divergent resposne across countries
        kappa = pm.Gamma("kappa", alpha=7.5, beta=1)
        

        # We assume that participants have a mean belief that is a function of their country
        # Note that we use a non-centered parameterization here to avoid divergent transitions
        participant_sigma = pm.HalfNormal("participant_sigma", sigma=sigma_participant)
        participant_offset = pm.Normal(
            "participant_offset", mu=0, sigma=1, dims="participants"
        )
        participant = pm.Deterministic('participant', 
                                    country[participant_country_idx] + participant_sigma * participant_offset, 
                                    dims='participants')

            
        alpha_theta = pm.Cauchy('alpha_theta', -2,2)
        beta_theta = pm.Cauchy('beta_theta', 0,2)
            
        #Estimated pre-treatment belief 
        belief_pre = participant[particpant_idx]  + item[item_idx]
        
        #Estimated post-treatment belief
        belief = belief_pre + u_t[treatment_idx, 0] + u_t[treatment_idx, 1] * belief_pre
        
        #Scaling factor for Tau

        #Additional code for posterior predictive simulation 

        y_sim_fixed_belief = pm.Deterministic('y_sim_fixed_belief', 
                                            sim_belief + u_t[treatment_sim, 0] + \
                                            u_t[treatment_sim, 1] * sim_belief)
        avg_belief = pm.Deterministic('avg_belief',pm.invlogit(sim_belief))
    

                                                
        y_out = ZOIBProportion(
            "y_out",
            mu=pm.invlogit(belief),
            kappa=kappa,
            theta=pm.invlogit(alpha_theta + beta_theta * pm.math.abs(belief)), 
            tau=pm.invlogit(belief),
            observed=y,
            shape=y.shape
        )
        
        return model, df, temp