#pragma once

#include "ofxsImageEffect.h"

/// OpenFX plugin factory registering BetterMotionBlur descriptors, parameters, and clean host unload.
class BetterMotionBlurPluginFactory : public OFX::PluginFactoryHelper<BetterMotionBlurPluginFactory>
{
public:
    BetterMotionBlurPluginFactory();
    virtual void load() override {}
    virtual void unload() override;
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context);
};
