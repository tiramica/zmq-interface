/*
 ------------------------------------------------------------------
 
 ZMQInterface
 Copyright (C) 2016 FP Battaglia
 
 based on
 Open Ephys GUI
 Copyright (C) 2013, 2015 Open Ephys
 
 ------------------------------------------------------------------
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 */

#include "ZmqInterfaceEditor.h"
#include "ZmqInterface.h"

class ZmqInterfaceEditor::ZmqInterfaceEditorListBox: public ListBox,
private ListBoxModel, public AsyncUpdater
{
public:
    
    ZmqInterfaceEditorListBox(const String noItemsText, ZmqInterfaceEditor *e):
        ListBox(String(), nullptr), noItemsMessage(noItemsText)
    {
        editor = e;
        setModel(this);
        
        backgroundGradient = ColourGradient(Colour(220, 220, 220), 0.0f, 0.0f,
                                            Colour(195, 195, 195), 0.0f, 120.0f, false);
        backgroundGradient.addColour(0.2f, Colour(185, 185, 185));
        
        backgroundColor = Colour(155, 155, 155);
        
        setColour(backgroundColourId, backgroundColor);
        
        refresh();

    }
    
    void handleAsyncUpdate()
    {
        refresh();
    }
    
    void refresh()
    {
        updateContent();
        repaint();
        
    }
    
    int getNumRows() override
    {
        return editor->getApplicationList()->size();
    }
    
    
    void paintListBoxItem (int row, Graphics& g, int width, int height, bool rowIsSelected) override
    {
        
        OwnedArray<ZmqApplication> *items = editor->getApplicationList();
        
        if (isPositiveAndBelow (row, items->size()))
        {
            g.fillAll(Colour(155, 155, 155));
            if (rowIsSelected)
                g.fillAll (findColour (TextEditor::highlightColourId)
                           .withMultipliedAlpha (0.3f));
            
            ZmqApplication *i = (*items)[row];
            const String item (i->name); // TODO change when we put a map
                
            const int x = getTickX();
            
            g.setFont (height * 0.7f);
            if (i->alive)
                g.setColour(Colours::green);
            else
                g.setColour(Colours::red);
            g.drawText (item, 10, 0, width - x - 2, height, Justification::centredLeft, true);
        } // end of function
    }
    
    
    void listBoxItemClicked (int row, const MouseEvent& e) override
    {
        selectRow (row);
    }
    
    void paintOverChildren (Graphics& g) override
    {
        if (editor->getApplicationList()->size() == 0)
        {
            g.setColour (Colours::darkgrey);
            g.setFont (14.0f);
        
            if (!CoreServices::getAcquisitionStatus())
            {
                g.drawText ("Waiting...",
                            10, 0, getWidth(), getHeight() / 2,
                            Justification::centredLeft, true);
            }
            else
            {
                g.drawText (noItemsMessage,
                        10, 0, getWidth(), getHeight() / 2,
                        Justification::centredLeft, true);
            }
        }
    }
    
private:
    const String noItemsMessage;
    ZmqInterfaceEditor *editor;
    /** Stores the editor's background color. */
    Colour backgroundColor;
    
    /** Stores the editor's background gradient. */
    ColourGradient backgroundGradient;
    
    int getTickX() const
    {
        return getRowHeight() + 5;
    }
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ZmqInterfaceEditorListBox)
};

ZmqInterfaceEditor::ZmqInterfaceEditor(GenericProcessor *parentNode): GenericEditor(parentNode)
{
    ZmqProcessor = (ZmqInterface *)parentNode;

    desiredWidth = 280;

    listBox = std::make_unique<ZmqInterfaceEditorListBox>(String("None"), this);
    listBox->setBounds(112,45,160,80);
    addAndMakeVisible(listBox.get());

    listTitle = std::make_unique<Label>("ListBox Label", "Connected apps:");
    listTitle->setColour(Label::textColourId, Colours::black);
    listTitle->setBounds(112,27,160,15);
    listTitle->setFont(Font("Fira Code", "SemiBold", 14.0f));
    addAndMakeVisible(listTitle.get());

    addComboBoxParameterEditor("Stream", 15, 22);
    parameterEditors.getLast()->setBounds(15, 22, 120, 42);
    
    // addSelectedChannelsParameterEditor("Channels", 10, 67);
    Parameter* maskChansParam = getProcessor()->getParameter("Channels");
    maskchannelsEditor = std::make_unique<MaskChannelsParameterEditor>(maskChansParam);
    maskchannelsEditor->setBounds(15, 67, maskchannelsEditor->getWidth(), maskchannelsEditor->getHeight());
    addAndMakeVisible(maskchannelsEditor.get());
    
    addTextBoxParameterEditor("data_port", 15, 87);
}

ZmqInterfaceEditor::~ZmqInterfaceEditor()
{

}

void ZmqInterfaceEditor::refreshListAsync()
{
    listBox->triggerAsyncUpdate();
}

void ZmqInterfaceEditor::startAcquisition()
{
    listBox->refresh();
}

void ZmqInterfaceEditor::stopAcquisition()
{
    listBox->refresh();
}

OwnedArray<ZmqApplication> *ZmqInterfaceEditor::getApplicationList()
{
    OwnedArray<ZmqApplication> *ar = ZmqProcessor->getApplicationList();
    return ar;
    
}

void ZmqInterfaceEditor::updateMaskChannelsParameter(Parameter* param)
{
    maskchannelsEditor->setParameter(param);
}