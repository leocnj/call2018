
The new data added for the second edition of the Spoken CALL Shared Task is divided into three files and contains an extra column in the metadata. This reflects the improved annotation process we were able to implement by leveraging the Shared Task 1 systems. The process worked as follows:

1. The new data was selected from the full data set using the same methodology as for Shared Task 1, consonant with the requirement that none of the Shared Task 2 speakers should be present in the Shared Task 1 data.

2. Speech data was processed through the two best recognisers from Shared Task 1 (University of Birmingham and Korea).

3. The two sets of ASR outputs were merged, and then cleaned up by human transcribers at Geneva. The transcriptions use conventions slightly adapted from a large Dutch corpus project, as follows:

*silence  no comprehensible speech
*v        foreign word
*z        mispronounced word
*a        incomplete word
*x        indistinct word
xxx/-xxx  unknown word/words/part of word
ggg       non-speech noise 

4. The cleaned-up transcriptions were processed through four of the best systems from Shared Task 1 (University of Birmingham, University of Pittsburgh, Korea, ETS) to give accept/reject decisions on the 'language' (fully acceptable) criterion.

5. Using the machine outputs, the data was divided into three preliminary groups (4-0, 3-1, 2-2), depending on well the machines agreed. The division was 70% 4-0, 22% 3-1, 8% 2-2.

6. Three English native speaker annotators familiar with the domain (two from Geneva, one from Cambridge) independently annotated 200 randomly chosen utterances from each group. On the 4-0 group, we found that the humans agreed about 98% with the machines, which was about as well as they agreed with each other. We consequently decided to consider this portion of the data as reliably judged by machine.

7. The three human annotators independently judged the remaining 3-1 and 2-2 portions of the data for both 'language' (full acceptability) and 'meaning' (meaning acceptability). To perform focussed checking for careless errors, the subsets where pairs of annotators differed were extracted and then independently rejudged by the annotators, together with an additional 20% control items where the annotators agreed. 

8. The machines were not set to provide meaning judgements, though if they marked an item as language=correct, this implied that it was also meaning=correct. The subset of the 4-0 group where the machines had unanimously rejected was extremely straightforward to judge, and we decided that it was enough for one human annotator to add meaning annotations for this portion of the data.

9. At the end of the annotation process, the material was divided into three bands by descending reliability, labelled as A, B and C:

  A. (5526 utterances) Either the machines are 4-0 and at least one human supports them, or the machines are 3-1 and all three humans support them.

  B. (873 utterances) All three humans agree, and at least one machine supports them.

  C. (299 utterances) Remaining cases. The 'language' accept/reject judgement is the majority human judgement.

10. The 'meaning' annotations were combined as follows. If the machines were unanimous and accepted, the item was marked as meaning=correct; otherwise the marking was the majority decision of the human judges.

The last column in the metadata spreadsheets summarises the machine and human judgements on 'language' (fully correct). So for example 'M: 3-1 H: 2-1' means '3 machine accepted and 1 rejected, 2 humans accepted and 1 rejected'.
